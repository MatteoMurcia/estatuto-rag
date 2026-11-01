"""Búsqueda híbrida: fusión de rankings con Reciprocal Rank Fusion (RRF)."""

from collections.abc import Sequence
from dataclasses import replace

from estatuto_rag.retrieval.base import Recuperador, Resultado

K_RRF = 60


def rrf(rankings: Sequence[list[Resultado]], k: int = K_RRF) -> list[Resultado]:
    """Fusiona rankings sumando 1 / (k + posición) de cada documento en cada lista.

    Solo usa la POSICIÓN, no la puntuación: así no hay que normalizar escalas tan distintas
    como la similitud coseno (0-1) y ts_rank_cd (sin techo).
    """
    puntos: dict[str, float] = {}
    primero_visto: dict[str, Resultado] = {}
    for ranking in rankings:
        for posicion, resultado in enumerate(ranking, start=1):
            puntos[resultado.id] = puntos.get(resultado.id, 0.0) + 1.0 / (k + posicion)
            primero_visto.setdefault(resultado.id, resultado)
    orden = sorted(puntos, key=lambda id_: (-puntos[id_], id_))
    return [replace(primero_visto[id_], score=puntos[id_]) for id_ in orden]


class RecuperadorHibrido:
    def __init__(self, recuperadores: Sequence[Recuperador], candidatos: int = 30) -> None:
        self._recuperadores = recuperadores
        self._candidatos = candidatos

    def recuperar(self, pregunta: str, k: int) -> list[Resultado]:
        rankings = [r.recuperar(pregunta, self._candidatos) for r in self._recuperadores]
        return rrf(rankings)[:k]
