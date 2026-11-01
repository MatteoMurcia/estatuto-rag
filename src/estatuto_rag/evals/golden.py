"""Carga del golden set (data/golden/questions.jsonl)."""

import json
from dataclasses import dataclass
from pathlib import Path

from estatuto_rag.ingest.pipeline import RAIZ

GOLDEN = RAIZ / "data" / "golden" / "questions.jsonl"
TIPOS = ("factual", "parafraseada", "multi", "sin_respuesta")


@dataclass(frozen=True)
class PreguntaGolden:
    id: str
    pregunta: str
    articulos_relevantes: list[str]
    respuesta_referencia: str
    tipo: str
    revisada: bool

    @property
    def tiene_respuesta(self) -> bool:
        return bool(self.articulos_relevantes)


def cargar_golden(ruta: Path = GOLDEN) -> list[PreguntaGolden]:
    with ruta.open(encoding="utf-8") as f:
        return [PreguntaGolden(**json.loads(linea)) for linea in f if linea.strip()]
