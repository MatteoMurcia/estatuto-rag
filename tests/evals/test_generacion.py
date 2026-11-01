"""Tests de la evaluación de generación y del juez, con LLMs falsos."""

import pytest

from estatuto_rag.evals.generacion import cita_acierta, evaluar_pregunta, resumir, tabla_markdown
from estatuto_rag.evals.golden import PreguntaGolden
from estatuto_rag.evals.juez import Juez, si_no
from estatuto_rag.generation.citas import ABSTENCION
from estatuto_rag.generation.llm import Mensaje
from estatuto_rag.retrieval.base import Resultado


class _Recuperador:
    def recuperar(self, pregunta: str, k: int) -> list[Resultado]:
        return [Resultado("38.1", "38", "1", "Artículo 38", "30 días naturales.", 1.0)]


class _LLM:
    def __init__(self, texto: str) -> None:
        self.texto = texto
        self.llamadas = 0

    def generar(self, mensajes: list[Mensaje]) -> str:
        self.llamadas += 1
        return self.texto


def _p(tipo: str = "factual", refs: list[str] | None = None) -> PreguntaGolden:
    refs = ["38"] if refs is None else refs
    return PreguntaGolden("x", "¿Cuántos días de vacaciones?", refs, "30 días.", tipo, True)


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [("SI", True), ("Sí.", True), ("**NO**", False), ("no, porque…", False), ("Quizá", None)],
)
def test_si_no(texto: str, esperado: bool | None) -> None:
    assert si_no(texto) is esperado


@pytest.mark.parametrize(
    ("cita", "ref", "esperado"),
    [("38.1", "38", True), ("38", "38.1", True), ("38.2", "38.1", False), ("3", "38", False)],
)
def test_cita_acierta(cita: str, ref: str, esperado: bool) -> None:
    assert cita_acierta(cita, ref) is esperado


def test_respuesta_con_cita_correcta_y_juez() -> None:
    fila = evaluar_pregunta(
        _p(), _Recuperador(), _LLM("Son 30 días [art. 38.1]."), Juez(_LLM("SI"))
    )
    assert fila.cita_correcta and not fila.citas_invalidas and not fila.abstencion
    assert fila.fiel is True and fila.correcta is True


def test_cita_inventada_se_detecta() -> None:
    fila = evaluar_pregunta(_p(), _Recuperador(), _LLM("Son 30 días [art. 99]."), None)
    assert fila.citas_invalidas == ["99"]
    assert not fila.cita_correcta


def test_abstencion_indebida_es_incorrecta_sin_llamar_al_juez() -> None:
    juez_llm = _LLM("SI")
    fila = evaluar_pregunta(_p(), _Recuperador(), _LLM(ABSTENCION), Juez(juez_llm))
    assert fila.abstencion and fila.correcta is False and fila.fiel is None
    assert juez_llm.llamadas == 0


def test_resumen_separa_preguntas_con_y_sin_respuesta() -> None:
    rec = _Recuperador()
    filas = [
        evaluar_pregunta(_p(), rec, _LLM("Son 30 días [art. 38.1]."), None),
        evaluar_pregunta(_p(), rec, _LLM(ABSTENCION), None),
        evaluar_pregunta(_p("sin_respuesta", []), rec, _LLM(ABSTENCION), None),
        evaluar_pregunta(_p("sin_respuesta", []), rec, _LLM("Sí [art. 7]."), None),
    ]
    resumen = resumir(filas)
    assert resumen["abstencion_correcta"] == 0.5
    assert resumen["abstencion_indebida"] == 0.5
    assert resumen["citas_inventadas"] == 0.25
    assert resumen["cita_correcta"] == 0.5
    assert resumen["fidelidad"] is None  # sin juez
    assert "| hibrido | 0.50 | 0.50 | 0.25 | 0.50 | - | - |" in tabla_markdown({"hibrido": resumen})
