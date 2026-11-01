"""Tipos comunes a todos los recuperadores."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Resultado:
    id: str
    articulo: str
    apartado: str | None
    encabezado: str
    texto: str
    score: float


class Recuperador(Protocol):
    def recuperar(self, pregunta: str, k: int) -> list[Resultado]: ...


COLUMNAS = "id, articulo, apartado, encabezado, texto"
