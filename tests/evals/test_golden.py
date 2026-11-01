"""Comprobaciones de integridad del golden set real."""

from estatuto_rag.evals.golden import TIPOS, cargar_golden
from estatuto_rag.ingest.pipeline import CHUNKS_JSONL, leer_jsonl


def test_golden_bien_formado() -> None:
    golden = cargar_golden()
    assert len(golden) == 40
    assert len({p.id for p in golden}) == 40
    assert all(p.tipo in TIPOS for p in golden)
    for p in golden:
        assert p.tiene_respuesta == (p.tipo != "sin_respuesta"), p.id


def test_referencias_existen_en_el_corpus() -> None:
    chunks = leer_jsonl(CHUNKS_JSONL)
    articulos = {c.articulo for c in chunks}
    apartados = {f"{c.articulo}.{c.apartado}" for c in chunks if c.apartado}
    # Un apartado puede no tener chunk propio si su artículo entero cabe en uno.
    enteros = {c.articulo for c in chunks if c.apartado is None}
    for p in cargar_golden():
        for ref in p.articulos_relevantes:
            art = ref.split(".")[0]
            assert art in articulos, f"{p.id}: artículo {art} no existe"
            if "." in ref:
                assert ref in apartados or art in enteros, f"{p.id}: {ref} no existe"
