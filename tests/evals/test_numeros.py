"""Tests de la fidelidad numérica."""

import pytest

from estatuto_rag.evals.numeros import numeros, numeros_no_respaldados


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("treinta y tres días", {33}),
        ("setecientos veinte días", {720}),
        ("Son 30 días naturales", {30}),
        ("doce horas y dieciséis años", {12, 16}),
        ("el 12 de febrero de 2012", {12, 2012}),
        ("dos mil", {2000}),
        ("Son 30 días [art. 38.1]", {30}),  # la cita no es un dato
        ("según el artículo 34.7, nueve horas", {9}),
    ],
)
def test_numeros(texto: str, esperado: set[int]) -> None:
    assert numeros(texto) == esperado


def test_detecta_la_cifra_inventada() -> None:
    contexto = "no podrá ser superior a nueve diarias ... mediarán, como mínimo, doce horas"
    respuesta = "La jornada no puede exceder de 42 horas y nueve diarias [art. 34.3]."
    assert numeros_no_respaldados(respuesta, contexto) == {42}


def test_cifra_en_letra_frente_a_cifra_en_numero() -> None:
    assert not numeros_no_respaldados("Son 30 días", "treinta días naturales")
    assert not numeros_no_respaldados("un año", "por año de servicio")
