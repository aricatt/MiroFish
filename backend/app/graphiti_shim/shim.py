"""GraphitiShim: drop-in replacement for the Zep Cloud SDK client.

This module implements a duck-typed adapter that mimics the subset of the
Zep Cloud SDK interface used by MiroFish. When GRAPH_BACKEND=graphiti,
get_zep_client() returns a GraphitiShim instance instead of a real Zep
client, so all existing MiroFish services work without modification.

Simulated interface:
    client.graph.create(graph_id, name, description)
    client.graph.delete(graph_id=...)
    client.graph.get(graph_id)
    client.graph.set_ontology(graph_ids, entities, edges)
    client.graph.add(graph_id, type, data, created_at, source_description, metadata)
    client.graph.search(graph_id, query, limit, scope, reranker)
    client.graph.episode.get(uuid_)
    client.graph.node.get(uuid_)
    client.graph.node.get_edges(node_uuid)
    client.graph.node.with_raw_response.get_by_graph_id(graph_id, limit, cursor)
    client.graph.edge.with_raw_response.get_by_graph_id(graph_id, limit, cursor)
    client.batch.create(metadata)
    client.batch.add(batch_id, items)
    client.batch.process(batch_id)
    client.batch.get(batch_id)
    client.batch.list(limit, cursor)
    client.batch.list_items(batch_id, limit, cursor)
"""

from __future__ import annotations

import time
import uuid as _uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from . import graph_manager
from ..utils.logger import get_logger

logger = get_logger('mirofish.graphiti_shim')

# Use zep_cloud's NotFoundError so existing `except NotFoundError` blocks
# in MiroFish services catch exceptions raised by the shim.
try:
    from zep_cloud import NotFoundError
except ImportError:
    class NotFoundError(Exception):
        """Fallback when zep_cloud is not installed."""


class _EpisodeClient:
    """Simulates client.graph.episode.get(uuid_)."""

    def get(self, *, uuid_: str) -> SimpleNamespace:
        # Graphiti processes episodes synchronously during add_episode,
        # so any existing episode is always "processed".
        return SimpleNamespace(uuid_=uuid_, processed=True)


class _NodeClient:
    """Simulates client.graph.node.get(uuid_) and .get_edges(node_uuid)."""

    def get(self, *, uuid_: str) -> SimpleNamespace | None:
        from graphiti_core.nodes import EntityNode as GEntityNode
        # Search all known graph instances — we don't know which graph this node belongs to
        for graph_id, g in graph_manager._graphiti_instances.items():
            try:
                node = graph_manager.run_async(GEntityNode.get_by_uuid(g.driver, uuid_))
                return _entity_node_to_namespace(node)
            except Exception:
                continue
        raise NotFoundError(f'Node {uuid_} not found')

    def get_edges(self, *, node_uuid: str) -> list[SimpleNamespace]:
        from graphiti_core.edges import EntityEdge as GEntityEdge
        for g in graph_manager._graphiti_instances.values():
            try:
                edges = graph_manager.run_async(
                    GEntityEdge.get_by_node_uuid(g.driver, node_uuid)
                )
                return [_entity_edge_to_namespace(e) for e in edges]
            except Exception:
                continue
        return []


class _RawResponseWrapper:
    """Simulates client.graph.node.with_raw_response.get_by_graph_id(graph_id, limit, cursor)."""

    def __init__(self, fetch_fn):
        self._fetch_fn = fetch_fn

    def get_by_graph_id(
        self,
        graph_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> SimpleNamespace:
        items = self._fetch_fn(graph_id)
        # Zep pagination: if we have more items than limit, slice and set cursor.
        # For simplicity, return all items in one page (no cursor header → stops).
        return SimpleNamespace(
            data=items,
            headers={},
        )


class _NodeNamespace:
    """Simulates client.graph.node."""

    def __init__(self):
        self.get = _NodeClient().get
        self.get_edges = _NodeClient().get_edges
        self.with_raw_response = _RawResponseWrapper(self._fetch_all_nodes)

    @staticmethod
    def _fetch_all_nodes(graph_id: str) -> list[SimpleNamespace]:
        from graphiti_core.nodes import EntityNode as GEntityNode
        graphiti = graph_manager.get_graphiti(graph_id)
        if graphiti is None:
            return []
        nodes = graph_manager.run_async(
            GEntityNode.get_by_group_ids(graphiti.driver, [graph_id])
        )
        return [_entity_node_to_namespace(n) for n in nodes]


class _EdgeNamespace:
    """Simulates client.graph.edge."""

    def __init__(self):
        self.with_raw_response = _RawResponseWrapper(self._fetch_all_edges)

    @staticmethod
    def _fetch_all_edges(graph_id: str) -> list[SimpleNamespace]:
        from graphiti_core.edges import EntityEdge as GEntityEdge
        graphiti = graph_manager.get_graphiti(graph_id)
        if graphiti is None:
            return []
        try:
            edges = graph_manager.run_async(
                GEntityEdge.get_by_group_ids(graphiti.driver, [graph_id])
            )
        except Exception:
            return []
        return [_entity_edge_to_namespace(e) for e in edges]


class _GraphSubClient:
    """Simulates client.graph.* (create, delete, get, set_ontology, add, search, episode, node, edge)."""

    def __init__(self):
        self.episode = _EpisodeClient()
        self.node = _NodeNamespace()
        self.edge = _EdgeNamespace()

    def create(
        self,
        *,
        graph_id: str,
        name: str,
        description: str = '',
    ) -> SimpleNamespace:
        graph_manager.get_or_create_graphiti(graph_id)
        logger.info('Graph created (Graphiti): graph_id=%s, name=%s', graph_id, name)
        return SimpleNamespace(graph_id=graph_id, name=name, description=description)

    def delete(self, *, graph_id: str) -> None:
        graph_manager.delete_graph(graph_id)
        logger.info('Graph deleted (Graphiti): graph_id=%s', graph_id)

    def get(self, graph_id: str) -> SimpleNamespace:
        graphiti = graph_manager.get_graphiti(graph_id)
        if graphiti is None:
            raise NotFoundError(f'Graph {graph_id} not found')
        return SimpleNamespace(graph_id=graph_id, name=graph_id)

    def set_ontology(
        self,
        *,
        graph_ids: list[str],
        entities: dict[str, type] | None = None,
        edges: dict[str, tuple] | None = None,
    ) -> None:
        # Convert Zep edge format to Graphiti format:
        # Zep:     edges = {name: (EdgeClass, [SourceTarget(source, target), ...])}
        # Graphiti: edge_types = {name: EdgeClass}
        #           edge_type_map = {(source_type, target_type): [edge_name, ...]}
        edge_types: dict[str, type] = {}
        edge_type_map: dict[tuple[str, str], list[str]] = {}
        if edges:
            for edge_name, (edge_class, source_targets) in edges.items():
                edge_types[edge_name] = edge_class
                for st in source_targets:
                    src = getattr(st, 'source', 'Entity')
                    tgt = getattr(st, 'target', 'Entity')
                    key = (src, tgt)
                    edge_type_map.setdefault(key, []).append(edge_name)

        for gid in graph_ids:
            graph_manager.cache_ontology(gid, {
                'entity_types': entities or {},
                'edge_types': edge_types or {},
                'edge_type_map': edge_type_map or {},
            })
        logger.info('Ontology cached for graphs: %s', graph_ids)

    def add(
        self,
        *,
        graph_id: str,
        type: str = 'text',
        data: str = '',
        created_at: str | None = None,
        source_description: str = '',
        metadata: dict[str, Any] | None = None,
    ) -> SimpleNamespace:
        graphiti = graph_manager.get_or_create_graphiti(graph_id)

        ref_time = datetime.now(timezone.utc)
        if created_at:
            try:
                ref_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass

        ontology = graph_manager.get_cached_ontology(graph_id) or {}
        entity_types = ontology.get('entity_types') or None
        edge_types = ontology.get('edge_types') or None
        edge_type_map = ontology.get('edge_type_map') or None

        from graphiti_core.nodes import EpisodeType
        episode_type = EpisodeType.message
        if type == 'text':
            episode_type = EpisodeType.text
        elif type == 'json':
            episode_type = EpisodeType.json

        result = graph_manager.run_async(
            graphiti.add_episode(
                name=f'mirofish_{graph_id}_{int(time.time())}',
                episode_body=data,
                source_description=source_description or 'MiroFish',
                reference_time=ref_time,
                source=episode_type,
                group_id=graph_id,
                entity_types=entity_types,
                edge_types=edge_types,
                edge_type_map=edge_type_map,
            )
        )

        episode_uuid = str(result.episode.uuid) if result.episode else str(_uuid.uuid4())
        return SimpleNamespace(
            uuid_=episode_uuid,
            uuid=episode_uuid,
            processed=True,
        )

    def search(
        self,
        *,
        graph_id: str | None = None,
        query: str = '',
        limit: int = 10,
        scope: str = 'edges',
        reranker: str = 'cross_encoder',
    ) -> SimpleNamespace:
        graphiti = graph_manager.get_graphiti(graph_id)
        if graphiti is None:
            return SimpleNamespace(edges=[], nodes=[])

        if scope == 'nodes':
            # Use search_() which returns SearchResults with both nodes and edges
            from graphiti_core.search.search_config import COMBINED_HYBRID_SEARCH_CROSS_ENCODER
            results = graph_manager.run_async(
                graphiti.search_(
                    query=query,
                    group_ids=[graph_id],
                    config=COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
                )
            )
            node_namespaces = [_entity_node_to_namespace(n) for n in results.nodes]
            return SimpleNamespace(edges=[], nodes=node_namespaces)
        else:
            # Default: search edges using the simple search() method
            edges = graph_manager.run_async(
                graphiti.search(
                    query=query,
                    group_ids=[graph_id],
                    num_results=limit,
                )
            )
            edge_namespaces = [_entity_edge_to_namespace(e) for e in edges]
            return SimpleNamespace(edges=edge_namespaces, nodes=[])


class _BatchClient:
    """Simulates client.batch.* — Graphiti processes episodes synchronously.

    The Zep Batch API flow is: create → add items → process → poll status.
    With Graphiti, add_episode is synchronous, so we buffer items and process
    them all at once on .process().
    """

    def __init__(self):
        self._batches: dict[str, dict[str, Any]] = {}
        self._graph_client = _GraphSubClient()

    def create(self, *, metadata: dict[str, Any] | None = None) -> SimpleNamespace:
        batch_id = str(_uuid.uuid4())
        self._batches[batch_id] = {
            'metadata': metadata or {},
            'items': [],
            'status': 'draft',
        }
        return SimpleNamespace(batch_id=batch_id)

    def add(self, *, batch_id: str, items: list[Any]) -> list[SimpleNamespace]:
        batch = self._batches.get(batch_id)
        if batch is None:
            raise ValueError(f'Batch {batch_id} not found')
        item_details = []
        for i, item in enumerate(items):
            episode_uuid = str(_uuid.uuid4())
            detail = SimpleNamespace(
                sequence_index=len(batch['items']) + i,
                episode_uuid=episode_uuid,
                status='pending',
            )
            item_details.append(detail)
            batch['items'].append({
                'item': item,
                'detail': detail,
            })
        return item_details

    def process(self, *, batch_id: str) -> None:
        batch = self._batches.get(batch_id)
        if batch is None:
            raise ValueError(f'Batch {batch_id} not found')

        for entry in batch['items']:
            item = entry['item']
            detail = entry['detail']
            try:
                graph_id = getattr(item, 'graph_id', None)
                data = getattr(item, 'data', '')
                data_type = getattr(item, 'data_type', 'text')
                source_description = getattr(item, 'source_description', 'MiroFish')
                metadata = getattr(item, 'metadata', {})

                if graph_id and data:
                    result = self._graph_client.add(
                        graph_id=graph_id,
                        type=data_type,
                        data=data,
                        source_description=source_description,
                        metadata=metadata,
                    )
                    detail.episode_uuid = result.uuid_
                detail.status = 'succeeded'
            except Exception as e:
                detail.status = 'failed'
                detail.error = str(e)
                logger.error('Batch item processing failed: %s', e)

        batch['status'] = 'succeeded'

    def get(self, *, batch_id: str) -> SimpleNamespace:
        batch = self._batches.get(batch_id)
        if batch is None:
            raise ValueError(f'Batch {batch_id} not found')
        succeeded = sum(1 for e in batch['items'] if e['detail'].status == 'succeeded')
        return SimpleNamespace(
            batch_id=batch_id,
            status=batch['status'],
            progress=SimpleNamespace(
                percent_complete=100 if batch['status'] == 'succeeded' else 0,
                succeeded_items=succeeded,
            ),
        )

    def list(self, *, limit: int = 100, cursor: int | None = None) -> SimpleNamespace:
        batches = []
        for batch_id, batch_data in self._batches.items():
            batches.append(SimpleNamespace(
                batch_id=batch_id,
                metadata=batch_data['metadata'],
                status=batch_data['status'],
            ))
        return SimpleNamespace(batches=batches[:limit], next_cursor=None)

    def list_items(
        self,
        *,
        batch_id: str,
        limit: int = 100,
        cursor: int | None = None,
    ) -> SimpleNamespace:
        batch = self._batches.get(batch_id)
        if batch is None:
            raise ValueError(f'Batch {batch_id} not found')
        items = [e['detail'] for e in batch['items']]
        return SimpleNamespace(items=items, next_cursor=None)


class GraphitiShim:
    """Drop-in replacement for zep_cloud.client.Zep.

    Exposes .graph and .batch sub-clients with the same method signatures
    that MiroFish services call on the real Zep SDK.
    """

    def __init__(self):
        self.graph = _GraphSubClient()
        self.batch = _BatchClient()
        logger.info('GraphitiShim initialized (GRAPH_BACKEND=graphiti)')


# ── Helpers ──────────────────────────────────────────────────────────

def _entity_node_to_namespace(node) -> SimpleNamespace:
    """Convert a Graphiti EntityNode to a Zep-like namespace object."""
    return SimpleNamespace(
        uuid_=node.uuid,
        uuid=node.uuid,
        name=node.name,
        labels=node.labels or [],
        summary=node.summary or '',
        attributes=node.attributes or {},
        created_at=node.created_at,
        group_id=node.group_id,
    )


def _entity_edge_to_namespace(edge) -> SimpleNamespace:
    """Convert a Graphiti EntityEdge to a Zep-like namespace object."""
    return SimpleNamespace(
        uuid_=edge.uuid,
        uuid=edge.uuid,
        name=edge.name or '',
        fact=edge.fact or '',
        source_node_uuid=edge.source_node_uuid,
        target_node_uuid=edge.target_node_uuid,
        created_at=edge.created_at,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        expired_at=edge.expired_at,
        episodes=edge.episodes or [],
        attributes=edge.attributes or {},
    )
