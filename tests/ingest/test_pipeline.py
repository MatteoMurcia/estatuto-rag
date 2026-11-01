"""Tests de la ingesta completa sobre el XML real."""

import json
from pathlib import Path

import pytest

from estatuto_rag.ingest.chunking import MAX_TOKENS, contar_tokens
from estatuto_rag.ingest.models import Chunk
from estatuto_rag.ingest.pipeline import XML_BOE, construir_chunks, guardar_jsonl


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return construir_chunks(XML_BOE.read_text(encoding="utf-8"))


def test_articulo_38_dice_treinta_dias_naturales(chunks: list[Chunk]) -> None:
    del_38 = [c for c in chunks if c.articulo == "38"]
    assert any("treinta días naturales" in c.texto for c in del_38)


def test_ids_unicos(chunks: list[Chunk]) -> None:
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_ningun_chunk_vacio_ni_demasiado_largo(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        assert chunk.texto.strip(), chunk.id
        assert contar_tokens(chunk.texto) <= MAX_TOKENS, chunk.id


def test_excluye_disposiciones_derogadas(chunks: list[Chunk]) -> None:
    # La DA 16.ª y la DA 21.ª solo contienen "(Derogada)".
    assert not {"da16", "da21"} & {c.articulo for c in chunks}


def test_guardar_jsonl(chunks: list[Chunk], tmp_path: Path) -> None:
    ruta = tmp_path / "chunks.jsonl"
    guardar_jsonl(chunks[:3], ruta)

    filas = [json.loads(linea) for linea in ruta.read_text(encoding="utf-8").splitlines()]
    assert [f["id"] for f in filas] == [c.id for c in chunks[:3]]
    assert filas[0]["texto_indexado"].startswith("Estatuto de los Trabajadores")
