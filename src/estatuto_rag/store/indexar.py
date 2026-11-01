"""Carga de chunks con sus embeddings en Postgres (idempotente)."""

from collections.abc import Iterator

import psycopg

from estatuto_rag.embeddings import Embedder
from estatuto_rag.ingest.models import Chunk

UPSERT = """
INSERT INTO chunks (id, articulo, apartado, padre, encabezado, texto, texto_indexado, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    articulo = EXCLUDED.articulo,
    apartado = EXCLUDED.apartado,
    padre = EXCLUDED.padre,
    encabezado = EXCLUDED.encabezado,
    texto = EXCLUDED.texto,
    texto_indexado = EXCLUDED.texto_indexado,
    embedding = EXCLUDED.embedding
"""


def _lotes(chunks: list[Chunk], tamano: int) -> Iterator[list[Chunk]]:
    for inicio in range(0, len(chunks), tamano):
        yield chunks[inicio : inicio + tamano]


def indexar(
    conn: psycopg.Connection, chunks: list[Chunk], embedder: Embedder, lote: int = 32
) -> int:
    """Calcula embeddings por lotes y hace upsert por `id`. Devuelve cuántos chunks cargó."""
    for grupo in _lotes(chunks, lote):
        vectores = embedder.embed([c.texto_indexado for c in grupo])
        filas = [
            (c.id, c.articulo, c.apartado, c.padre, c.encabezado, c.texto, c.texto_indexado, v)
            for c, v in zip(grupo, vectores, strict=True)
        ]
        with conn.cursor() as cur:
            cur.executemany(UPSERT, filas)
    return len(chunks)
