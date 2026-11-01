"""Grafo del agente.

    START → reescribir → buscar → responder ─┬─► END
                ▲                             │
                └──── se abstuvo y quedan ────┘
                      intentos (MAX_INTENTOS)

Diferencias con el RAG simple (generation/responder.py):
1. Busca con la pregunta original + 1-3 reescrituras en lenguaje jurídico (RRF entre ellas).
2. Si el modelo se abstiene, no se rinde a la primera: vuelve a reescribir con términos
   distintos de los ya probados y busca otra vez.
El prompt de respuesta y la validación de citas son los mismos, para que la comparación
mida solo el efecto de la reescritura y el reintento.
"""

import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from estatuto_rag.generation.llm import LLM
from estatuto_rag.generation.responder import Respuesta, generar_respuesta
from estatuto_rag.retrieval.base import Recuperador, Resultado
from estatuto_rag.retrieval.reescritura import Reescritor, recuperar_multiconsulta

MAX_INTENTOS = 2


class Estado(TypedDict):
    pregunta: str
    inicio: float
    intentos: int
    consultas: list[str]  # todas las probadas hasta ahora (la original incluida)
    nuevas: list[str]  # las de este intento
    fuentes: list[Resultado]
    respuesta: Respuesta | None


# Nota: LangGraph exige que el parámetro de cada nodo se llame `state`.


class Agente:
    def __init__(
        self, recuperador: Recuperador, llm: LLM, reescritor: Reescritor, k: int = 5
    ) -> None:
        self._recuperador = recuperador
        self._llm = llm
        self._reescritor = reescritor
        self._k = k
        self._grafo = self._construir()

    def _construir(self) -> CompiledStateGraph[Estado, None, Estado, Estado]:
        grafo = StateGraph(Estado)
        grafo.add_node("reescribir", self._reescribir)
        grafo.add_node("buscar", self._buscar)
        grafo.add_node("responder", self._responder)
        grafo.add_edge(START, "reescribir")
        grafo.add_edge("reescribir", "buscar")
        grafo.add_edge("buscar", "responder")
        grafo.add_conditional_edges("responder", self._siguiente, ["reescribir", END])
        return grafo.compile()

    def _reescribir(self, state: Estado) -> dict[str, object]:
        previas = state["consultas"][1:]  # la original no cuenta como intento fallido
        nuevas = self._reescritor.reescribir(state["pregunta"], previas)
        return {
            "intentos": state["intentos"] + 1,
            "nuevas": nuevas,
            "consultas": state["consultas"] + nuevas,
        }

    def _buscar(self, state: Estado) -> dict[str, object]:
        # En el primer intento, la original va con las reescrituras; en el reintento, solo
        # las nuevas: repetir la original volvería a traer el mismo contexto que falló.
        consultas = state["nuevas"] or [state["pregunta"]]
        if state["intentos"] == 1:
            consultas = [state["pregunta"], *state["nuevas"]]
        return {"fuentes": recuperar_multiconsulta(self._recuperador, consultas, self._k)}

    def _responder(self, state: Estado) -> dict[str, object]:
        respuesta = generar_respuesta(
            state["pregunta"],
            state["fuentes"],
            self._llm,
            state["inicio"],
            consultas=state["consultas"],
        )
        return {"respuesta": respuesta}

    def _siguiente(self, state: Estado) -> str:
        respuesta = state["respuesta"]
        if respuesta is not None and respuesta.abstencion and state["intentos"] < MAX_INTENTOS:
            return "reescribir"
        return END

    def responder(self, pregunta: str) -> Respuesta:
        inicial: Estado = {
            "pregunta": pregunta,
            "inicio": time.perf_counter(),
            "intentos": 0,
            "consultas": [pregunta],
            "nuevas": [],
            "fuentes": [],
            "respuesta": None,
        }
        final = self._grafo.invoke(inicial)
        respuesta: Respuesta = final["respuesta"]
        return respuesta
