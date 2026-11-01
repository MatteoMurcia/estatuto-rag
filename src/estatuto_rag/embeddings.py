"""Cliente de embeddings: bge-m3 servido por Ollama."""

from typing import Protocol

import httpx

DIMENSION = 1024  # bge-m3 devuelve vectores de 1024 dimensiones


class Embedder(Protocol):
    """Cualquier cosa que convierta textos en vectores. Permite usar un falso en los tests."""

    def embed(self, textos: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    def __init__(self, url: str, modelo: str, timeout: float = 120.0) -> None:
        self._cliente = httpx.Client(base_url=url, timeout=timeout)
        self._modelo = modelo

    def embed(self, textos: list[str]) -> list[list[float]]:
        respuesta = self._cliente.post("/api/embed", json={"model": self._modelo, "input": textos})
        respuesta.raise_for_status()
        vectores: list[list[float]] = respuesta.json()["embeddings"]
        return vectores
