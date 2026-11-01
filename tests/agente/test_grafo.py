"""Tests del grafo del agente, con LLMs falsos (sin Ollama)."""

from estatuto_rag.agente.grafo import MAX_INTENTOS, Agente
from estatuto_rag.generation.citas import ABSTENCION
from estatuto_rag.generation.llm import Mensaje
from estatuto_rag.retrieval.base import Resultado
from estatuto_rag.retrieval.reescritura import Reescritor


class LLMSecuencia:
    """Devuelve las respuestas en orden y repite la última."""

    def __init__(self, *respuestas: str) -> None:
        self.respuestas = list(respuestas)
        self.llamadas = 0

    def generar(self, mensajes: list[Mensaje]) -> str:
        texto = self.respuestas[min(self.llamadas, len(self.respuestas) - 1)]
        self.llamadas += 1
        return texto


class RecuperadorEco:
    """Cada consulta devuelve un chunk cuyo id es la propia consulta: así se ve qué se buscó."""

    def __init__(self) -> None:
        self.consultas: list[str] = []

    def recuperar(self, pregunta: str, k: int) -> list[Resultado]:
        self.consultas.append(pregunta)
        return [Resultado(pregunta, pregunta, None, "", "", 0.0)]


def test_responde_a_la_primera_sin_reescribir() -> None:
    recuperador = RecuperadorEco()
    generador = LLMSecuencia("Son 30 días [art. 38].")
    reescritor_llm = LLMSecuencia("vacaciones anuales")
    agente = Agente(recuperador, generador, Reescritor(reescritor_llm))
    r = agente.responder("¿cuántas vacaciones tengo?")
    assert not r.abstencion
    assert generador.llamadas == 1
    assert reescritor_llm.llamadas == 0  # sin abstención no se paga la reescritura
    assert recuperador.consultas == ["¿cuántas vacaciones tengo?"]
    assert r.consultas == ["¿cuántas vacaciones tengo?"]


def test_si_se_abstiene_reescribe_y_busca_solo_con_las_reescrituras() -> None:
    recuperador = RecuperadorEco()
    generador = LLMSecuencia(ABSTENCION, "Es despido disciplinario [art. 54].")
    reescritor = Reescritor(LLMSecuencia("despido disciplinario\nfaltas de puntualidad"))
    r = Agente(recuperador, generador, reescritor).responder("me han echado por llegar tarde")

    assert not r.abstencion
    assert generador.llamadas == 2
    assert recuperador.consultas == [
        "me han echado por llegar tarde",
        "despido disciplinario",
        "faltas de puntualidad",
    ]
    assert r.consultas == recuperador.consultas


def test_se_rinde_tras_el_maximo_de_intentos() -> None:
    generador = LLMSecuencia(ABSTENCION)
    reescritor = Reescritor(LLMSecuencia("consulta a", "consulta b", "consulta c"))
    r = Agente(RecuperadorEco(), generador, reescritor).responder("¿cuánto es el paro?")
    assert r.abstencion
    assert generador.llamadas == MAX_INTENTOS
