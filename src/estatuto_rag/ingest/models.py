"""Modelos de datos de la ingesta.

`Articulo` es lo que sale del parser del BOE (un precepto: artículo o disposición).
`Chunk` es la unidad que se indexa en Postgres y que se cita en las respuestas.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Articulo:
    """Un precepto del Estatuto tal como lo devuelve el parser.

    `numero` es el identificador que se cita: "38", "52", "da1" (disposición adicional 1)…
    `texto` lleva los apartados separados por saltos de línea; cada apartado numerado empieza
    por "1. ", "2. "…
    """

    numero: str
    titulo: str
    texto: str
    titulo_ley: str | None = None  # "Título I"
    capitulo: str | None = None  # "Capítulo II"
    seccion: str | None = None  # "Sección 5.ª"


@dataclass(frozen=True)
class Chunk:
    """Fragmento indexable.

    - `id`: "38" si el artículo entra entero, "38.1", "38.2"… si se trocea por apartados.
    - `padre`: el número del artículo cuando el chunk es un apartado; None si es el artículo entero.
    - `encabezado`: la ruta jerárquica que se antepone al texto (contextual retrieval barato).
    """

    id: str
    articulo: str
    apartado: str | None
    padre: str | None
    encabezado: str
    texto: str

    @property
    def texto_indexado(self) -> str:
        """Lo que se embebe y se indexa en full-text: encabezado + texto."""
        return f"{self.encabezado}\n{self.texto}"
