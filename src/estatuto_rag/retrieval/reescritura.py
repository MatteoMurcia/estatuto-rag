"""Reescritura de la consulta: de lenguaje coloquial a vocabulario del Estatuto.

"Me han echado por llegar tarde" no comparte palabras ni apenas significado con "faltas
repetidas e injustificadas de puntualidad" (art. 54.2.a). Un LLM reformula la pregunta en
términos jurídicos y se busca con todas las versiones a la vez, fusionadas con RRF.
"""

import re
from collections.abc import Sequence

from estatuto_rag.generation.llm import LLM, Mensaje
from estatuto_rag.retrieval.base import Recuperador, Resultado
from estatuto_rag.retrieval.hibrido import rrf

MAX_CONSULTAS = 3

SISTEMA = """Conviertes preguntas de trabajadores en consultas de búsqueda sobre el Estatuto \
de los Trabajadores de España.

Escribe con el vocabulario jurídico que usa la ley: por ejemplo "despido disciplinario", \
"extinción del contrato", "excedencia", "permiso retribuido", "movilidad geográfica", \
"periodo de prueba", "jornada", "salario".

Devuelve de 1 a 3 consultas cortas, una por línea, sin numerar, sin comillas y sin explicar nada."""

REINTENTO = """Estas consultas no encontraron la respuesta: {previas}
Propón consultas con términos jurídicos distintos."""

_VINETA = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def parsear_consultas(texto: str, excluir: Sequence[str] = ()) -> list[str]:
    """Una consulta por línea, sin viñetas, comillas ni repetidas (ni las ya probadas)."""
    vistas = {e.lower() for e in excluir}
    consultas: list[str] = []
    for linea in texto.splitlines():
        consulta = _VINETA.sub("", linea).strip().strip("\"'«»").strip()
        if consulta and consulta.lower() not in vistas:
            vistas.add(consulta.lower())
            consultas.append(consulta)
    return consultas[:MAX_CONSULTAS]


class Reescritor:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def reescribir(self, pregunta: str, previas: Sequence[str] = ()) -> list[str]:
        contenido = pregunta
        if previas:
            contenido += "\n\n" + REINTENTO.format(previas="; ".join(previas))
        mensajes: list[Mensaje] = [
            {"role": "system", "content": SISTEMA},
            {"role": "user", "content": contenido},
        ]
        return parsear_consultas(self._llm.generar(mensajes), excluir=[pregunta, *previas])


def recuperar_multiconsulta(
    base: Recuperador, consultas: Sequence[str], k: int, candidatos: int = 30
) -> list[Resultado]:
    """Busca con cada consulta y fusiona los rankings con RRF."""
    return rrf([base.recuperar(c, candidatos) for c in consultas])[:k]


class RecuperadorConReescritura:
    """Recuperador que busca con la pregunta original más sus reescrituras."""

    def __init__(self, base: Recuperador, reescritor: Reescritor, candidatos: int = 30) -> None:
        self._base = base
        self._reescritor = reescritor
        self._candidatos = candidatos

    def recuperar(self, pregunta: str, k: int) -> list[Resultado]:
        consultas = [pregunta, *self._reescritor.reescribir(pregunta)]
        return recuperar_multiconsulta(self._base, consultas, k, self._candidatos)
