"""Ollama embedding adapter for Graphiti's EmbedderClient interface.

Uses Ollama's OpenAI-compatible /v1/embeddings endpoint so we can leverage
the standard AsyncOpenAI client. This avoids a custom HTTP layer.
"""

from __future__ import annotations

from collections.abc import Iterable

from openai import AsyncOpenAI

from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig

from ..config import Config


class OllamaEmbedderConfig(EmbedderConfig):
    """Configuration for the Ollama embedder."""

    def __init__(self, **kwargs):
        super().__init__(
            embedding_dim=kwargs.pop('embedding_dim', Config.OLLAMA_EMBED_DIM),
        )
        # Store as object attributes (not Pydantic fields, since EmbedderConfig is frozen)
        object.__setattr__(self, 'embedding_model', kwargs.get('embedding_model', Config.OLLAMA_EMBED_MODEL))
        object.__setattr__(self, 'api_key', kwargs.get('api_key', 'ollama'))
        object.__setattr__(self, 'base_url', kwargs.get('base_url', Config.OLLAMA_BASE_URL + '/v1'))


class OllamaEmbedder(EmbedderClient):
    """Embedder that talks to Ollama's OpenAI-compatible embeddings endpoint."""

    def __init__(self, config: OllamaEmbedderConfig | None = None):
        if config is None:
            config = OllamaEmbedderConfig()
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        result = await self.client.embeddings.create(
            input=input_data, model=self.config.embedding_model
        )
        return result.data[0].embedding[: self.config.embedding_dim]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        result = await self.client.embeddings.create(
            input=input_data_list, model=self.config.embedding_model
        )
        return [e.embedding[: self.config.embedding_dim] for e in result.data]
