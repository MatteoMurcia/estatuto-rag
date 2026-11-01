"""Fixtures compartidas."""

import hashlib
import math

import psycopg
import pytest

from estatuto_rag.config import cargar_config
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


@pytest.fixture(scope="session")
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture(scope="session")
def database_url() -> str:
    """BD de tests separada (`estatuto_test`), para no vaciar la que usas en desarrollo."""
    base = cargar_config().database_url
    url_test = base.rsplit("/", 1)[0] + "/estatuto_test"
    with psycopg.connect(base, autocommit=True) as conn:
        existe = conn.execute("SELECT 1 FROM pg_database WHERE datname = 'estatuto_test'")
        if existe.fetchone() is None:
            conn.execute("CREATE DATABASE estatuto_test")
    return url_test
