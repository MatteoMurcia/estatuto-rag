"""Estrategias de recuperación que se comparan en las evaluaciones."""

import psycopg

from estatuto_rag.embeddings import Embedder
from estatuto_rag.retrieval.base import Recuperador
from estatuto_rag.retrieval.hibrido import RecuperadorHibrido
from estatuto_rag.retrieval.textual import RecuperadorTextual
from estatuto_rag.retrieval.vectorial import RecuperadorVectorial

NOMBRES = ("vector", "texto", "hibrido", "hibrido+rerank")


def construir_estrategias(
    conn: psycopg.Connection, embedder: Embedder, rerank: bool
) -> dict[str, Recuperador]:
    vectorial = RecuperadorVectorial(conn, embedder)
    textual = RecuperadorTextual(conn)
    hibrido = RecuperadorHibrido([vectorial, textual])
    estrategias: dict[str, Recuperador] = {
        "vector": vectorial,
        "texto": textual,
        "hibrido": hibrido,
    }
    if rerank:
        # Import diferido: sentence-transformers es un extra opcional (uv sync --extra rerank).
        from estatuto_rag.retrieval.reranker import CrossEncoderPuntuador, RecuperadorConReranker

        estrategias["hibrido+rerank"] = RecuperadorConReranker(hibrido, CrossEncoderPuntuador())
    return estrategias
