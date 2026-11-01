"""Tests de integración de los recuperadores sobre el Estatuto completo (con FakeEmbedder)."""

from collections.abc import Iterator

import psycopg
import pytest

from estatuto_rag.embeddings import Embedder
from estatuto_rag.ingest.pipeline import CHUNKS_JSONL, leer_jsonl
from estatuto_rag.retrieval.hibrido import RecuperadorHibrido
from estatuto_rag.retrieval.textual import RecuperadorTextual
from estatuto_rag.retrieval.vectorial import RecuperadorVectorial
from estatuto_rag.store.db import conectar
from estatuto_rag.store.indexar import indexar

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder(fake_embedder: Embedder) -> Embedder:
    return fake_embedder


@pytest.fixture(scope="module")
def conn(database_url: str, embedder: Embedder) -> Iterator[psycopg.Connection]:
    with conectar(database_url) as conexion:
        conexion.execute("TRUNCATE chunks")
        indexar(conexion, leer_jsonl(CHUNKS_JSONL), embedder)
        yield conexion
        conexion.execute("TRUNCATE chunks")


def test_textual_encuentra_vacaciones_con_pregunta_natural(conn: psycopg.Connection) -> None:
    resultados = RecuperadorTextual(conn).recuperar("¿Cuántos días de vacaciones me tocan?", 5)

    assert "38" in [r.id for r in resultados]


def test_textual_sin_coincidencias_devuelve_vacio(conn: psycopg.Connection) -> None:
    assert RecuperadorTextual(conn).recuperar("xyzzy plugh", 5) == []


def test_vectorial_devuelve_k_resultados_ordenados(
    conn: psycopg.Connection, embedder: Embedder
) -> None:
    resultados = RecuperadorVectorial(conn, embedder).recuperar("periodo de prueba", 5)

    assert len(resultados) == 5
    scores = [r.score for r in resultados]
    assert scores == sorted(scores, reverse=True)


def test_hibrido_combina_ambos(conn: psycopg.Connection, embedder: Embedder) -> None:
    hibrido = RecuperadorHibrido([RecuperadorVectorial(conn, embedder), RecuperadorTextual(conn)])

    resultados = hibrido.recuperar("vacaciones anuales retribuidas", 5)

    assert len(resultados) == 5
    assert resultados[0].articulo == "38"
