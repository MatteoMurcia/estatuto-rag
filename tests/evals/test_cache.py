"""Tests de la caché de embeddings."""

from pathlib import Path

import pytest

from estatuto_rag.evals.cache import EmbedderConCache, cargar_cache, clave, guardar_cache


class _Contador:
    def __init__(self) -> None:
        self.llamadas: list[list[str]] = []

    def embed(self, textos: list[str]) -> list[list[float]]:
        self.llamadas.append(textos)
        return [[float(len(t)), 0.5] for t in textos]


def test_guardar_y_cargar_ida_y_vuelta(tmp_path: Path) -> None:
    ruta = tmp_path / "cache.jsonl.gz"
    guardar_cache({clave("a"): [0.1234567891, 1.0]}, ruta)
    assert cargar_cache(ruta) == {clave("a"): [0.123457, 1.0]}


def test_guardar_es_determinista(tmp_path: Path) -> None:
    vectores = {clave("b"): [1.0], clave("a"): [2.0]}
    guardar_cache(vectores, tmp_path / "1.gz")
    guardar_cache(dict(reversed(vectores.items())), tmp_path / "2.gz")
    assert (tmp_path / "1.gz").read_bytes() == (tmp_path / "2.gz").read_bytes()


def test_cache_inexistente_es_vacia(tmp_path: Path) -> None:
    assert cargar_cache(tmp_path / "no.gz") == {}


def test_solo_pide_al_respaldo_lo_que_falta() -> None:
    respaldo = _Contador()
    embedder = EmbedderConCache({clave("hola"): [9.0, 9.0]}, respaldo)
    assert embedder.embed(["hola", "adiós"]) == [[9.0, 9.0], [5.0, 0.5]]
    assert respaldo.llamadas == [["adiós"]]
    embedder.embed(["adiós"])
    assert len(respaldo.llamadas) == 1  # el segundo ya estaba en caché


def test_sin_respaldo_falla_con_mensaje_util() -> None:
    with pytest.raises(KeyError, match=r"python -m estatuto_rag\.evals\.cache"):
        EmbedderConCache({}).embed(["texto nuevo"])
