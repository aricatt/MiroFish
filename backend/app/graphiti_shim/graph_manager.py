"""Manage {graph_id → Graphiti instance} mapping.

Each Zep "graph" maps to a Graphiti instance with a corresponding Neo4j
group_id. We create the Graphiti instance lazily on first access and cache
it for the process lifetime.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from ..config import Config
from .ollama_embedder import OllamaEmbedder, OllamaEmbedderConfig
from ..utils.logger import get_logger

logger = get_logger('mirofish.graphiti_shim.graph_manager')

_graphiti_instances: dict[str, Graphiti] = {}
_lock = threading.RLock()
_event_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_ontology_cache: dict[str, dict[str, Any]] = {}


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Return a shared background event loop for async Graphiti calls."""
    global _event_loop, _loop_thread
    if _event_loop is not None and not _event_loop.is_closed():
        return _event_loop
    with _lock:
        if _event_loop is not None and not _event_loop.is_closed():
            return _event_loop
        _event_loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(
            target=_event_loop.run_forever,
            daemon=True,
            name='graphiti-async-loop',
        )
        _loop_thread.start()
    return _event_loop


def _run_async(coro: Any) -> Any:
    """Run a coroutine on the shared background loop and return its result."""
    loop = _get_or_create_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def _create_llm_client():
    """Create an OpenAI-compatible LLM client pointing at Ollama."""
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    config = LLMConfig(
        api_key='ollama',
        model=Config.OLLAMA_MODEL,
        base_url=Config.OLLAMA_BASE_URL + '/v1',
        max_tokens=16384,
    )
    return OpenAIGenericClient(
        config=config,
        structured_output_mode='json_object',
    )


def _create_embedder() -> OllamaEmbedder:
    return OllamaEmbedder(OllamaEmbedderConfig())


def get_or_create_graphiti(graph_id: str):
    """Get an existing Graphiti instance or create a new one for graph_id."""
    if graph_id in _graphiti_instances:
        return _graphiti_instances[graph_id]
    with _lock:
        if graph_id in _graphiti_instances:
            return _graphiti_instances[graph_id]
        logger.info('Creating Graphiti instance for graph_id=%s', graph_id)
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.llm_client.config import LLMConfig
        llm_config = LLMConfig(
            api_key='ollama',
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL + '/v1',
        )
        graphiti = Graphiti(
            uri=Config.NEO4J_URI,
            user=Config.NEO4J_USER,
            password=Config.NEO4J_PASSWORD,
            llm_client=_create_llm_client(),
            embedder=_create_embedder(),
            cross_encoder=OpenAIRerankerClient(config=llm_config),
        )
        _run_async(graphiti.build_indices_and_constraints())
        _graphiti_instances[graph_id] = graphiti
    return _graphiti_instances[graph_id]


def get_graphiti(graph_id: str):
    """Return the cached Graphiti instance or None."""
    return _graphiti_instances.get(graph_id)


def delete_graph(graph_id: str) -> None:
    """Delete a graph: remove from cache and delete all nodes by group_id."""
    graphiti = _graphiti_instances.pop(graph_id, None)
    if graphiti is not None:
        from graphiti_core.nodes import Node
        _run_async(Node.delete_by_group_id(graphiti.driver, graph_id))
        _run_async(graphiti.close())
    _ontology_cache.pop(graph_id, None)


def cache_ontology(graph_id: str, ontology: dict[str, Any]) -> None:
    """Cache ontology entity_types/edge_types for later use in add_episode."""
    _ontology_cache[graph_id] = ontology


def get_cached_ontology(graph_id: str) -> dict[str, Any] | None:
    return _ontology_cache.get(graph_id)


def run_async(coro: Any) -> Any:
    """Public wrapper for running async code from sync context."""
    return _run_async(coro)
