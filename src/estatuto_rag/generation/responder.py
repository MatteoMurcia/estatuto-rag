"""RAG simple: recuperar -> generar -> validar citas."""

import time
from dataclasses import dataclass

from estatuto_rag.generation.citas import es_abstencion, extraer_citas, id_cita
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


def responder(pregunta: str, recuperador: Recuperador, llm: LLM, k: int = 5) -> Respuesta:
    inicio = time.perf_counter()
    fuentes = recuperador.recuperar(pregunta, k)
    texto = llm.generar(construir_mensajes(pregunta, fuentes))

    permitidas = {id_cita(f) for f in fuentes} | {f.articulo for f in fuentes}
    citas = extraer_citas(texto)
    return Respuesta(
        pregunta=pregunta,
        respuesta=texto,
        citas=[c for c in citas if c in permitidas],
        citas_invalidas=[c for c in citas if c not in permitidas],
        abstencion=es_abstencion(texto),
        fuentes=fuentes,
        latencia_ms=(time.perf_counter() - inicio) * 1000,
    )
