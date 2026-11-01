"""Chunking por artículo, con troceo por apartados si el artículo es largo.

Te toca implementar estas tres funciones hasta que pasen `tests/ingest/test_chunking.py`.
"""

from estatuto_rag.ingest.models import Articulo, Chunk

MAX_TOKENS = 400


def contar_tokens(texto: str) -> int:
    """Número aproximado de tokens de `texto`.

    Decide tú la aproximación y justifícala (palabras x factor, tiktoken, el tokenizer de bge-m3…).
    """
    raise NotImplementedError


def encabezado(articulo: Articulo) -> str:
    """Ruta jerárquica del artículo.

    Ejemplo: "Estatuto de los Trabajadores · Título I · Capítulo II · Art. 38. Vacaciones anuales".
    Los niveles que sean None no aparecen.
    """
    raise NotImplementedError


def trocear_articulo(articulo: Articulo, max_tokens: int = MAX_TOKENS) -> list[Chunk]:
    """Un chunk por artículo; si supera `max_tokens`, un chunk por apartado numerado.

    Decisión abierta (apúntala en decisions.md): ¿qué haces si un apartado solo ya supera
    `max_tokens`, o si un artículo largo no tiene apartados numerados?
    """
    raise NotImplementedError
