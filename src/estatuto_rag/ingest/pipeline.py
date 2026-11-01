"""Pipeline de ingesta: XML del BOE -> artículos -> chunks -> data/chunks.jsonl."""

import json
from dataclasses import asdict
from pathlib import Path

from estatuto_rag.ingest.chunking import trocear_articulo
from estatuto_rag.ingest.models import Articulo, Chunk
from estatuto_rag.ingest.parser import parsear_boe

RAIZ = Path(__file__).resolve().parents[3]
XML_BOE = RAIZ / "data" / "raw" / "BOE-A-2015-11430_texto_2026-10-29.xml"
CHUNKS_JSONL = RAIZ / "data" / "chunks.jsonl"


def esta_derogado(articulo: Articulo) -> bool:
    """Un precepto cuyo único texto es "(Derogada)" no aporta nada al índice."""
    return articulo.texto.strip() in {"(Derogada)", "(Derogado)"}


def construir_chunks(xml: str) -> list[Chunk]:
    articulos = [a for a in parsear_boe(xml) if not esta_derogado(a)]
    return [chunk for articulo in articulos for chunk in trocear_articulo(articulo)]


def guardar_jsonl(chunks: list[Chunk], ruta: Path) -> None:
    with ruta.open("w", encoding="utf-8", newline="\n") as f:
        for chunk in chunks:
            fila = asdict(chunk) | {"texto_indexado": chunk.texto_indexado}
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")


def leer_jsonl(ruta: Path) -> list[Chunk]:
    chunks = []
    with ruta.open(encoding="utf-8") as f:
        for linea in f:
            fila = json.loads(linea)
            fila.pop("texto_indexado")  # es una propiedad calculada, no un campo
            chunks.append(Chunk(**fila))
    return chunks
