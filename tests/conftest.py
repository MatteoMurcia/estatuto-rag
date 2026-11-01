"""Fixtures compartidas."""

import hashlib
import math

import pytest

from estatuto_rag.embeddings import DIMENSION


class FakeEmbedder:
    """Embeddings deterministas sin Ollama: cada palabra activa una dimensión fija."""

    def embed(self, textos: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in textos]

    @staticmethod
    def _vector(texto: str) -> list[float]:
        v = [0.0] * DIMENSION
        for palabra in texto.lower().split():
            indice = int(hashlib.md5(palabra.encode()).hexdigest(), 16) % DIMENSION
            v[indice] += 1.0
        norma = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norma for x in v]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
