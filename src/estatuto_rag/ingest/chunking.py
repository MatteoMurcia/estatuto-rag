"""Chunking por artículo, con troceo por apartados si el artículo es largo.

Te toca implementar estas tres funciones hasta que pasen `tests/ingest/test_chunking.py`.
"""

import re

from estatuto_rag.ingest.models import Articulo, Chunk

MAX_TOKENS = 400
TOKENS_POR_PALABRA = 1.3

# Un apartado numerado empieza por "1. ", "2. "… al principio de la línea.
INICIO_APARTADO = re.compile(r"^(\d+)\. ")


def contar_tokens(texto: str) -> int:
    """Número aproximado de tokens de `texto`.

    Decide tú la aproximación y justifícala (palabras x factor, tiktoken, el tokenizer de bge-m3…).
    """
    palabras = len(texto.split())
    return round(palabras * TOKENS_POR_PALABRA)


def encabezado(articulo: Articulo) -> str:
    """Ruta jerárquica del artículo.

    Ejemplo: "Estatuto de los Trabajadores · Título I · Capítulo II · Art. 38. Vacaciones anuales".
    Los niveles que sean None no aparecen.
    """
    partes = ["Estatuto de los Trabajadores"]
    for nivel in (articulo.titulo_ley, articulo.capitulo, articulo.seccion):
        if nivel is not None:
            partes.append(nivel)
    partes.append(f"Art. {articulo.numero}. {articulo.titulo}")
    return " · ".join(partes)


def _separar_apartados(texto: str) -> list[tuple[str | None, str]]:
    """Agrupa las líneas en apartados: [(número o None, texto), ...]."""
    apartados: list[tuple[str | None, list[str]]] = []
    for linea in texto.split("\n"):
        coincidencia = INICIO_APARTADO.match(linea)
        if coincidencia:
            apartados.append((coincidencia.group(1), [linea]))
        elif apartados:
            apartados[-1][1].append(linea)
        else:
            apartados.append((None, [linea]))
    return [(numero, "\n".join(lineas)) for numero, lineas in apartados]


def trocear_articulo(articulo: Articulo, max_tokens: int = MAX_TOKENS) -> list[Chunk]:
    """Un chunk por artículo; si supera `max_tokens`, un chunk por apartado numerado.

    Decisión abierta (apúntala en decisions.md): ¿qué haces si un apartado solo ya supera
    `max_tokens`, o si un artículo largo no tiene apartados numerados?
    """
    cabecera = encabezado(articulo)

    if contar_tokens(articulo.texto) <= max_tokens:
        return [
            Chunk(
                id=articulo.numero,
                articulo=articulo.numero,
                apartado=None,
                padre=None,
                encabezado=cabecera,
                texto=articulo.texto,
            )
        ]

    chunks = []
    for numero, texto in _separar_apartados(articulo.texto):
        apartado = numero or "0"  # la introducción sin número se cita como "52.0"
        chunks.append(
            Chunk(
                id=f"{articulo.numero}.{apartado}",
                articulo=articulo.numero,
                apartado=apartado,
                padre=articulo.numero,
                encabezado=cabecera,
                texto=texto,
            )
        )
    return chunks
