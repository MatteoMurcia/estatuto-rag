"""Chunking por artículo, con troceo por apartados si el artículo es largo.

Te toca implementar estas tres funciones hasta que pasen `tests/ingest/test_chunking.py`.
"""

from estatuto_rag.ingest.models import Articulo, Chunk

MAX_TOKENS = 400
TOKENS_POR_PALABRA = 1.3


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


def trocear_articulo(articulo: Articulo, max_tokens: int = MAX_TOKENS) -> list[Chunk]:
    """Un chunk por artículo; si supera `max_tokens`, un chunk por apartado numerado.

    Decisión abierta (apúntala en decisions.md): ¿qué haces si un apartado solo ya supera
    `max_tokens`, o si un artículo largo no tiene apartados numerados?
    """
    raise NotImplementedError
