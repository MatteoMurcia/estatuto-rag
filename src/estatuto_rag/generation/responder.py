"""RAG simple: recuperar -> generar -> validar citas."""

import time
from dataclasses import dataclass, field

from estatuto_rag.generation.citas import cita_valida, es_abstencion, extraer_citas
from estatuto_rag.generation.llm import LLM
from estatuto_rag.generation.prompt import construir_mensajes
from estatuto_rag.retrieval.base import Recuperador, Resultado


@dataclass(frozen=True)
class Respuesta:
    pregunta: str
    respuesta: str
    citas: list[str]
    citas_invalidas: list[str]  # citadas por el LLM pero que no estaban en el contexto
    abstencion: bool
    fuentes: list[Resultado]
    latencia_ms: float
    consultas: list[str] = field(default_factory=list)  # búsquedas hechas (el agente hace varias)


def responder(pregunta: str, recuperador: Recuperador, llm: LLM, k: int = 5) -> Respuesta:
    inicio = time.perf_counter()
    fuentes = recuperador.recuperar(pregunta, k)
    return generar_respuesta(pregunta, fuentes, llm, inicio, consultas=[pregunta])


def generar_respuesta(
    pregunta: str,
    fuentes: list[Resultado],
    llm: LLM,
    inicio: float,
    consultas: list[str] | None = None,
) -> Respuesta:
    """Genera con el contexto ya recuperado y valida las citas. Lo comparten RAG y agente."""
    texto = llm.generar(construir_mensajes(pregunta, fuentes))
    citas = extraer_citas(texto)
    return Respuesta(
        pregunta=pregunta,
        respuesta=texto,
        citas=[c for c in citas if cita_valida(c, fuentes)],
        citas_invalidas=[c for c in citas if not cita_valida(c, fuentes)],
        abstencion=es_abstencion(texto),
        fuentes=fuentes,
        latencia_ms=(time.perf_counter() - inicio) * 1000,
        consultas=consultas or [pregunta],
    )
