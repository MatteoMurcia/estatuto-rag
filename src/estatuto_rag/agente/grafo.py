"""Grafo del agente.

    START → buscar → responder ─┬─► END
              ▲                 │
              └── reescribir ◄──┘  si se abstuvo y quedan intentos (MAX_INTENTOS)

1. Primer intento: igual que el RAG simple (busca con la pregunta tal cual, sin llamar al LLM
   para reescribir).
2. Solo si el modelo se abstiene, reescribe la pregunta en vocabulario jurídico y vuelve a
   buscar con las reescrituras.

¿Por qué no reescribir siempre? Medido sobre el golden set: reescribir en todas las preguntas
bajó el hit@5 de 0.85 a 0.79 (rompía preguntas directas que ya se encontraban) y no mejoró
las coloquiales. El valor está en el reintento, así que la reescritura es el plan B.
El prompt de respuesta y la validación de citas son los del RAG simple, para que la comparación
mida solo el efecto del reintento.
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
    intentos: int  # búsquedas hechas
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
        grafo.add_node("buscar", self._buscar)
        grafo.add_node("responder", self._responder)
        grafo.add_node("reescribir", self._reescribir)
        grafo.add_edge(START, "buscar")
        grafo.add_edge("buscar", "responder")
        grafo.add_conditional_edges("responder", self._siguiente, ["reescribir", END])
        grafo.add_edge("reescribir", "buscar")
        return grafo.compile()

    def _buscar(self, state: Estado) -> dict[str, object]:
        # Primer intento: la pregunta tal cual. Reintentos: solo las reescrituras nuevas
        # (repetir la original traería el mismo contexto que ya falló).
        consultas = state["nuevas"] or [state["pregunta"]]
        return {
            "intentos": state["intentos"] + 1,
            "fuentes": recuperar_multiconsulta(self._recuperador, consultas, self._k),
        }

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

    def _reescribir(self, state: Estado) -> dict[str, object]:
        previas = state["consultas"][1:]  # reescrituras ya probadas (la original va aparte)
        nuevas = self._reescritor.reescribir(state["pregunta"], previas)
        return {"nuevas": nuevas, "consultas": state["consultas"] + nuevas}

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
