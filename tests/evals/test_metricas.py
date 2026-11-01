"""Tests de las métricas de recuperación."""

import pytest

from estatuto_rag.evals.metricas import acierta, hit_at_k, recall_at_k, reciprocal_rank
from estatuto_rag.retrieval.base import Resultado


def _r(id_: str) -> Resultado:
    articulo, _, apartado = id_.split("#")[0].partition(".")
    return Resultado(
        id=id_, articulo=articulo, apartado=apartado or None, encabezado="", texto="", score=0
    )


@pytest.mark.parametrize(
    ("chunk", "referencia", "esperado"),
    [
        ("38", "38", True),  # artículo entero = referencia de artículo
        ("38", "38.1", True),  # el chunk del artículo entero contiene el 38.1
        ("14.1", "14.1", True),
        ("14.1", "14", True),  # referencia de artículo: vale cualquier apartado
        ("41.4#2", "41.4", True),  # las partes cuentan como su apartado
        ("14.3", "14.1", False),  # otro apartado del mismo artículo
        ("15.1", "14", False),
        ("140", "14", False),  # no basta con que empiece igual
    ],
)
def test_acierta(chunk: str, referencia: str, esperado: bool) -> None:
    assert acierta(_r(chunk), referencia) is esperado


def test_hit_at_k() -> None:
    resultados = [_r("1"), _r("2"), _r("38")]
    assert hit_at_k(resultados, ["38"], 3) == 1.0
    assert hit_at_k(resultados, ["38"], 2) == 0.0


def test_recall_at_k_en_preguntas_multi() -> None:
    resultados = [_r("53.1"), _r("2"), _r("3")]
    assert recall_at_k(resultados, ["53.1", "56.1"], 5) == 0.5


def test_reciprocal_rank() -> None:
    assert reciprocal_rank([_r("1"), _r("38")], ["38"]) == 0.5
    assert reciprocal_rank([_r("1")], ["38"]) == 0.0
