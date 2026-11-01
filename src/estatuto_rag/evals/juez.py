"""LLM como juez (phi4-mini en local): fidelidad al contexto y corrección frente a la referencia.

Cada criterio es una pregunta de sí/no por separado: a un modelo pequeño le cuesta menos
contestar una cosa concreta que rellenar una rúbrica entera, y "SI"/"NO" se parsea sin ambigüedad.
"""

import unicodedata

from estatuto_rag.generation.citas import formatear_contexto
from estatuto_rag.generation.llm import LLM, Mensaje
from estatuto_rag.retrieval.base import Resultado

SISTEMA_JUEZ = (
    "Eres un evaluador estricto de un sistema de preguntas y respuestas sobre el Estatuto de "
    "los Trabajadores. Contesta SOLO con una palabra: SI o NO."
)

FIDELIDAD = """<contexto>
{contexto}
</contexto>

<respuesta>
{respuesta}
</respuesta>

¿Está TODO lo que afirma la respuesta respaldado por el contexto? Si la respuesta añade \
algún dato (plazo, cantidad, condición) que no aparece en el contexto, contesta NO."""

CORRECCION = """Pregunta: {pregunta}

Respuesta de referencia (correcta): {referencia}

Respuesta a evaluar: {respuesta}

¿La respuesta a evaluar dice lo mismo que la referencia en lo esencial (mismos plazos, \
cantidades y condiciones), aunque use otras palabras? Si contradice la referencia, se \
deja lo esencial o dice que no encuentra la respuesta, contesta NO."""


def si_no(texto: str) -> bool | None:
    """'Sí.', 'SI', 'no, porque…' → True/False. Cualquier otra cosa → None (veredicto inválido)."""
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    palabras = sin_tildes.strip().upper().split()
    primera = palabras[0].strip(".,:;!*\"'") if palabras else ""
    if primera == "SI":
        return True
    if primera == "NO":
        return False
    return None


class Juez:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def _preguntar(self, contenido: str) -> bool | None:
        mensajes: list[Mensaje] = [
            {"role": "system", "content": SISTEMA_JUEZ},
            {"role": "user", "content": contenido},
        ]
        return si_no(self._llm.generar(mensajes))

    def es_fiel(self, respuesta: str, fuentes: list[Resultado]) -> bool | None:
        contexto = formatear_contexto(fuentes)
        return self._preguntar(FIDELIDAD.format(contexto=contexto, respuesta=respuesta))

    def es_correcta(self, pregunta: str, referencia: str, respuesta: str) -> bool | None:
        return self._preguntar(
            CORRECCION.format(pregunta=pregunta, referencia=referencia, respuesta=respuesta)
        )
