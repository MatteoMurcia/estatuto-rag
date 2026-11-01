"""Métricas de recuperación: hit@k, recall@k y MRR.

Un resultado "acierta" una referencia del golden set si cita lo mismo:
- referencia "38.1" ← chunk 38.1 (o sus partes 38.1#2), o el chunk "38" si el artículo va entero;
- referencia "38"   ← cualquier chunk del artículo 38.
"""

from collections.abc import Sequence

from estatuto_rag.generation.citas import id_cita
from estatuto_rag.retrieval.base import Resultado


def acierta(resultado: Resultado, referencia: str) -> bool:
    articulo_ref = referencia.split(".")[0]
    if resultado.articulo != articulo_ref:
        return False
    if "." not in referencia or resultado.apartado is None:
        return True
    return id_cita(resultado) == referencia


def es_relevante(resultado: Resultado, referencias: Sequence[str]) -> bool:
    return any(acierta(resultado, ref) for ref in referencias)


def hit_at_k(resultados: Sequence[Resultado], referencias: Sequence[str], k: int) -> float:
    """1 si algún resultado del top k es relevante; 0 si no."""
    return float(any(es_relevante(r, referencias) for r in resultados[:k]))


def recall_at_k(resultados: Sequence[Resultado], referencias: Sequence[str], k: int) -> float:
    """Fracción de referencias cubiertas por el top k (importa en preguntas multi-artículo)."""
    if not referencias:
        return 0.0
    cubiertas = sum(any(acierta(r, ref) for r in resultados[:k]) for ref in referencias)
    return cubiertas / len(referencias)


def reciprocal_rank(resultados: Sequence[Resultado], referencias: Sequence[str]) -> float:
    """1 / posición del primer resultado relevante (0 si no hay). MRR es su media."""
    for posicion, resultado in enumerate(resultados, start=1):
        if es_relevante(resultado, referencias):
            return 1.0 / posicion
    return 0.0
