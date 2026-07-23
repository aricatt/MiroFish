"""Graphiti shim package: drop-in replacement for the Zep Cloud SDK interface.

When Config.GRAPH_BACKEND == "graphiti", get_zep_client() returns a
GraphitiShim instance that mimics the Zep Cloud SDK's client.graph.* and
client.batch.* interface, delegating to a local Graphiti + Neo4j + Ollama
stack.
"""
