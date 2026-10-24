"""Local embedding backends for memory storage and retrieval."""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_DEFAULT_EMBEDDING_DIM = 384


@runtime_checkable
class Embedder(Protocol):
    """Protocol for text embedding models."""

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of produced vectors."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents.

        Args:
            texts: Document strings to embed.

        Returns:
            One embedding vector per input text.
        """

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query.

        Args:
            text: Query string.

        Returns:
            Query embedding vector.
        """


class SentenceTransformerEmbedder:
    """GPU/CPU embedder backed by sentence-transformers."""

    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        """Initialize the embedder lazily.

        Args:
            model_name: Hugging Face sentence-transformers model id.
            device: Torch device string. Auto-selects CUDA when available.
        """
        self._model_name = model_name
        self._device = device
        self._model = None
        self._embedding_dim: int | None = None

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensionality."""
        if self._embedding_dim is None:
            self._embedding_dim = self._get_model().get_sentence_embedding_dimension()
        return self._embedding_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document strings."""
        if not texts:
            return []
        vectors = self._get_model().encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string."""
        return self.embed_documents([text])[0]

    def _get_model(self):
        import torch
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            resolved_device = self._device
            if not resolved_device:
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "Loading memory embedding model %s on %s",
                self._model_name,
                resolved_device,
            )
            self._model = SentenceTransformer(self._model_name, device=resolved_device)
        return self._model


class DeterministicEmbedder:
    """Hash-based embedder for fast unit tests without torch."""

    def __init__(self, *, embedding_dim: int = _DEFAULT_EMBEDDING_DIM) -> None:
        """Initialize deterministic vectors.

        Args:
            embedding_dim: Size of generated embedding vectors.
        """
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensionality."""
        return self._embedding_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using stable SHA-256 derived vectors."""
        return [self._vector_for_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query using the same deterministic scheme."""
        return self._vector_for_text(text)

    def _vector_for_text(self, text: str) -> list[float]:
        values = [0.0] * self._embedding_dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._embedding_dim
            values[index] += 1.0
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            return values
        return [value / norm for value in values]


def build_embedder(model_name: str, *, device: str | None = None) -> Embedder:
    """Create the configured production embedder.

    Args:
        model_name: Hugging Face sentence-transformers model id.
        device: Optional torch device override.

    Returns:
        Sentence-transformers embedder instance.
    """
    return SentenceTransformerEmbedder(model_name, device=device or None)
