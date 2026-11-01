"""Búsqueda semántica: vecinos más cercanos por distancia coseno con pgvector."""

import psycopg

from estatuto_rag.embeddings import Embedder
from estatuto_rag.retrieval.base import COLUMNAS, Resultado

SQL = f"""
SELECT {COLUMNAS}, 1 - (embedding <=> %(vector)s::vector) AS score
FROM chunks
ORDER BY embedding <=> %(vector)s::vector
LIMIT %(k)s
"""


class RecuperadorVectorial:
    def __init__(self, conn: psycopg.Connection, embedder: Embedder) -> None:
        self._conn = conn
        self._embedder = embedder

    def recuperar(self, pregunta: str, k: int) -> list[Resultado]:
        [vector] = self._embedder.embed([pregunta])
        filas = self._conn.execute(SQL, {"vector": vector, "k": k}).fetchall()
        return [Resultado(*fila) for fila in filas]
