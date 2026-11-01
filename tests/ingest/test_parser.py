"""Tests del parser del BOE, contra XML real descargado en el spike (2026-10-29)."""

from pathlib import Path

import pytest

from estatuto_rag.ingest.models import Articulo
from estatuto_rag.ingest.parser import parsear_boe

RAIZ = Path(__file__).resolve().parents[2]
FIXTURE_A38 = RAIZ / "tests" / "fixtures" / "boe_a38.xml"
TEXTO_COMPLETO = RAIZ / "data" / "raw" / "BOE-A-2015-11430_texto_2026-10-29.xml"


@pytest.fixture(scope="module")
def estatuto() -> dict[str, Articulo]:
    articulos = parsear_boe(TEXTO_COMPLETO.read_text(encoding="utf-8"))
    return {a.numero: a for a in articulos}


# --- Un bloque suelto ------------------------------------------------------------------------


def test_bloque_a38() -> None:
    [articulo] = parsear_boe(FIXTURE_A38.read_text(encoding="utf-8"))

    assert articulo.numero == "38"
    assert articulo.titulo == "Vacaciones anuales"
    lineas = articulo.texto.split("\n")
    assert len(lineas) == 6  # 6 <p class="parrafo">, uno por línea
    assert lineas[0].startswith("1. El periodo de vacaciones")
    assert "treinta días naturales" in articulo.texto


# --- Texto completo --------------------------------------------------------------------------


def test_estan_todos_los_articulos(estatuto: dict[str, Articulo]) -> None:
    numericos = [n for n in estatuto if n.isdigit()]
    assert len(numericos) == 92


def test_jerarquia_del_articulo_38(estatuto: dict[str, Articulo]) -> None:
    a38 = estatuto["38"]
    assert a38.titulo_ley == "Título I"
    assert a38.capitulo == "Capítulo II"
    assert a38.seccion is not None and a38.seccion.startswith("Sección 5.ª")


def test_disposicion_adicional_primera(estatuto: dict[str, Articulo]) -> None:
    da1 = estatuto["da1"]
    assert da1.titulo == "Trabajo por cuenta propia"
    # Las disposiciones no heredan el Título/Capítulo/Sección del último artículo.
    assert da1.titulo_ley is None
    assert da1.capitulo is None
    assert da1.seccion is None


def test_excluye_los_bloques_del_real_decreto(estatuto: dict[str, Articulo]) -> None:
    assert not {"preambulo", "aunico", "ddunica", "firma"} & estatuto.keys()


def test_usa_solo_la_ultima_version(estatuto: dict[str, Articulo]) -> None:
    # El art. 48 tiene 8 versiones en el XML; su texto no puede repetirse.
    assert estatuto["48"].texto.count("Al cesar las causas legales de suspensión") == 1


def test_descarta_las_notas_al_pie(estatuto: dict[str, Articulo]) -> None:
    assert not any("Ref. BOE-" in a.texto for a in estatuto.values())
