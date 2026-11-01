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


def test_responde_a_la_primera_sin_reintentar() -> None:
    generador = LLMSecuencia("Son 30 días [art. 38].")
    agente = Agente(RecuperadorEco(), generador, Reescritor(LLMSecuencia("vacaciones anuales")))
    r = agente.responder("¿cuántas vacaciones tengo?")
    assert not r.abstencion
    assert generador.llamadas == 1
    assert r.consultas == ["¿cuántas vacaciones tengo?", "vacaciones anuales"]


def test_reintenta_con_consultas_nuevas_si_se_abstiene() -> None:
    recuperador = RecuperadorEco()
    generador = LLMSecuencia(ABSTENCION, "Es despido disciplinario [art. 54].")
    reescritor = Reescritor(LLMSecuencia("faltas de puntualidad", "despido disciplinario"))
    r = Agente(recuperador, generador, reescritor).responder("me han echado por llegar tarde")

    assert not r.abstencion and r.citas == []  # "54" no estaba en este contexto falso
    assert generador.llamadas == 2
    # Primer intento: original + reescritura. Reintento: solo la consulta nueva.
    assert recuperador.consultas == [
        "me han echado por llegar tarde",
        "faltas de puntualidad",
        "despido disciplinario",
    ]


def test_se_rinde_tras_el_maximo_de_intentos() -> None:
    generador = LLMSecuencia(ABSTENCION)
    reescritor = Reescritor(LLMSecuencia("consulta a", "consulta b", "consulta c"))
    r = Agente(RecuperadorEco(), generador, reescritor).responder("¿cuánto es el paro?")
    assert r.abstencion
    assert generador.llamadas == MAX_INTENTOS
