"""Tests de la reescritura de consultas, con LLM y recuperador falsos."""

from estatuto_rag.generation.llm import Mensaje
from estatuto_rag.retrieval.base import Resultado
from estatuto_rag.retrieval.reescritura import (
    RecuperadorConReescritura,
    Reescritor,
    parsear_consultas,
)


class LLMFijo:
    def __init__(self, texto: str) -> None:
        self.texto = texto
        self.mensajes: list[Mensaje] = []

    def generar(self, mensajes: list[Mensaje]) -> str:
        self.mensajes = mensajes
        return self.texto


def _r(id_: str) -> Resultado:
    return Resultado(id_, id_.split(".")[0], None, "", "", 0.0)


class RecuperadorPorPalabra:
    """Solo encuentra el art. 54 si la consulta dice 'despido'."""

    def recuperar(self, pregunta: str, k: int) -> list[Resultado]:
        if "despido" in pregunta:
            return [_r("54"), _r("55")]
        return [_r("34"), _r("37")]


def test_parsear_consultas_limpia_vinetas_comillas_y_repetidas() -> None:
    texto = '1. despido disciplinario\n- "faltas de puntualidad"\n\n• Despido disciplinario\nx\ny'
    assert parsear_consultas(texto) == ["despido disciplinario", "faltas de puntualidad", "x"]


def test_parsear_consultas_excluye_las_ya_probadas() -> None:
    assert parsear_consultas("despido\nextinción", excluir=["Despido"]) == ["extinción"]


def test_el_reintento_incluye_las_consultas_fallidas() -> None:
    llm = LLMFijo("extinción del contrato")
    Reescritor(llm).reescribir("¿me pueden echar?", previas=["despido disciplinario"])
    assert "despido disciplinario" in llm.mensajes[-1]["content"]


def test_la_reescritura_encuentra_lo_que_la_original_no() -> None:
    base = RecuperadorPorPalabra()
    pregunta = "Me han echado por llegar tarde"
    assert "54" not in [r.id for r in base.recuperar(pregunta, 5)]

    recuperador = RecuperadorConReescritura(base, Reescritor(LLMFijo("despido disciplinario")))
    ids = [r.id for r in recuperador.recuperar(pregunta, 5)]
    assert "54" in ids
    assert set(ids) == {"54", "55", "34", "37"}  # RRF conserva lo de la original
