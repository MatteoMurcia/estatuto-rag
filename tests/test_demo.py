"""Test de humo de la demo (se salta si no está instalado el extra `demo`)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from estatuto_rag.demo import resaltar_citas

DEMO = Path(__file__).parents[1] / "src" / "estatuto_rag" / "demo.py"


def test_resaltar_citas() -> None:
    assert resaltar_citas("Son 30 días [art. 38.1].") == "Son 30 días **:blue[[art. 38.1]]**."


def test_la_demo_arranca_sin_pregunta_y_sin_tocar_bd_ni_ollama() -> None:
    at = AppTest.from_file(str(DEMO)).run()
    assert not at.exception
    assert at.title[0].value == "Pregunta al Estatuto de los Trabajadores"
    assert len(at.sidebar.button) == 4  # los ejemplos
