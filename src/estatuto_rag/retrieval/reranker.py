"""Reranking con un cross-encoder: recupera muchos candidatos y reordena los mejores."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Protocol

from estatuto_rag.retrieval.base import Recuperador, Resultado

MODELO_RERANKER = "BAAI/bge-reranker-v2-m3"


class Puntuador(Protocol):
    def puntuar(self, pregunta: str, textos: Sequence[str]) -> list[float]: ...


class CrossEncoderPuntuador:
    """bge-reranker-v2-m3 con sentence-transformers (instalar con `uv sync --extra rerank`)."""

    def __init__(self, modelo: str = MODELO_RERANKER, max_length: int = 512) -> None:
        # Import diferido: el paquete solo se carga si de verdad se usa el reranker.
        from sentence_transformers import CrossEncoder

        self._modelo: Any = CrossEncoder(modelo, max_length=max_length)

    def puntuar(self, pregunta: str, textos: Sequence[str]) -> list[float]:
        pares = [(pregunta, texto) for texto in textos]
        return [float(s) for s in self._modelo.predict(pares)]


class RecuperadorConReranker:
    def __init__(self, base: Recuperador, puntuador: Puntuador, candidatos: int = 30) -> None:
        self._base = base
        self._puntuador = puntuador
        self._candidatos = candidatos

    def recuperar(self, pregunta: str, k: int) -> list[Resultado]:
        candidatos = self._base.recuperar(pregunta, self._candidatos)
        if not candidatos:
            return []
        textos = [f"{c.encabezado}\n{c.texto}" for c in candidatos]
        puntos = self._puntuador.puntuar(pregunta, textos)
        reordenados = sorted(
            zip(candidatos, puntos, strict=True), key=lambda par: par[1], reverse=True
        )
        return [replace(c, score=p) for c, p in reordenados[:k]]
