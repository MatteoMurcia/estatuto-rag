"""Tests unitarios de la fusión RRF (sin base de datos)."""

import pytest

from estatuto_rag.retrieval.base import Resultado
from estatuto_rag.retrieval.hibrido import rrf


def _r(id_: str, score: float = 0.0) -> Resultado:
    return Resultado(id=id_, articulo=id_, apartado=None, encabezado="", texto="", score=score)


def test_documento_en_ambas_listas_gana() -> None:
    vector = [_r("a"), _r("b"), _r("c")]
    texto = [_r("c"), _r("d")]

    fusion = rrf([vector, texto])

    # "c" es 3.º y 1.º: 1/63 + 1/61 > 1/61 de "a", que solo aparece una vez.
    assert fusion[0].id == "c"
    assert {r.id for r in fusion} == {"a", "b", "c", "d"}


def test_score_es_la_suma_de_reciprocos() -> None:
    fusion = rrf([[_r("a")], [_r("a")]], k=60)

    assert fusion[0].score == pytest.approx(2 / 61)


def test_ignora_las_puntuaciones_originales() -> None:
    # Escalas muy distintas (0.9 frente a 50): solo cuenta la posición.
    fusion = rrf([[_r("a", 0.9), _r("b", 0.1)], [_r("b", 50.0), _r("a", 1.0)]])

    assert fusion[0].score == pytest.approx(fusion[1].score)


def test_rankings_vacios() -> None:
    assert rrf([[], []]) == []
