"""Prompt del RAG simple."""

from estatuto_rag.generation.citas import ABSTENCION, formatear_contexto
from estatuto_rag.generation.llm import Mensaje
from estatuto_rag.retrieval.base import Resultado

SISTEMA = f"""Eres un asistente que responde preguntas sobre el Estatuto de los Trabajadores \
de España usando EXCLUSIVAMENTE los artículos que se te dan entre etiquetas <articulo>.

Reglas:
1. Responde en español, en 2 o 3 frases, de forma clara para alguien sin formación jurídica.
2. Después de cada afirmación, cita el artículo del que sale con el formato [art. ID], usando \
el atributo id de la etiqueta. Ejemplo: "Son 30 días naturales [art. 38.1]."
3. No cites nada que no esté en los artículos dados. No uses conocimiento propio.
4. Si los artículos no contienen la respuesta, responde exactamente: "{ABSTENCION}"
"""


def construir_mensajes(pregunta: str, resultados: list[Resultado]) -> list[Mensaje]:
    contexto = formatear_contexto(resultados)
    return [
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": f"{contexto}\n\nPregunta: {pregunta}"},
    ]
