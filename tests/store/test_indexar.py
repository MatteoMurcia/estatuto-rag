"""Tests de integración: necesitan Postgres con pgvector (docker compose up -d db)."""

from collections.abc import Iterator

import psycopg
import pytest

from estatuto_rag.config import cargar_config
from estatuto_rag.embeddings import Embedder
from estatuto_rag.ingest.models import Chunk
from estatuto_rag.store.db import conectar
from estatuto_rag.store.indexar import indexar

pytestmark = pytest.mark.integration


@pytest.fixture
def conn() -> Iterator[psycopg.Connection]:
    with conectar(cargar_config().database_url) as conexion:
        conexion.execute("TRUNCATE chunks")
        yield conexion
        conexion.execute("TRUNCATE chunks")


def _chunk(id_: str, texto: str) -> Chunk:
    return Chunk(
        id=id_,
        articulo=id_.split(".")[0],
        apartado=None,
        padre=None,
        encabezado=f"Estatuto de los Trabajadores · Art. {id_}",
        texto=texto,
    )


CHUNKS = [
    _chunk("38", "El periodo de vacaciones anuales retribuidas no será inferior a treinta días."),
    _chunk("14", "Podrá concertarse por escrito un periodo de prueba."),
]


def test_indexar_es_idempotente(conn: psycopg.Connection, fake_embedder: Embedder) -> None:
    indexar(conn, CHUNKS, fake_embedder)
    indexar(conn, CHUNKS, fake_embedder)

    [(total,)] = conn.execute("SELECT count(*) FROM chunks").fetchall()
    assert total == 2


def test_tsv_se_genera_con_stemming_en_espanol(
    conn: psycopg.Connection, fake_embedder: Embedder
) -> None:
    indexar(conn, CHUNKS, fake_embedder)

    # "vacación" (singular) encuentra "vacaciones" gracias al diccionario 'spanish'.
    filas = conn.execute(
        "SELECT id FROM chunks WHERE tsv @@ websearch_to_tsquery('spanish', 'vacación')"
    ).fetchall()
    assert filas == [("38",)]
