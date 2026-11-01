"""Tests del chunking. Escritos antes de la implementación (TDD): empiezan en rojo."""

import pytest

from estatuto_rag.ingest.chunking import (
    MAX_TOKENS,
    contar_tokens,
    encabezado,
    nombre_precepto,
    trocear_articulo,
)
from estatuto_rag.ingest.models import Articulo


def _apartado(n: int, palabras: int) -> str:
    return f"{n}. " + " ".join(["trabajador"] * palabras)


@pytest.fixture
def articulo_corto() -> Articulo:
    return Articulo(
        numero="38",
        titulo="Vacaciones anuales",
        texto=(
            "1. El periodo de vacaciones anuales retribuidas, no sustituible por compensación "
            "económica, será el pactado en convenio colectivo o contrato individual. En ningún "
            "caso la duración será inferior a treinta días naturales.\n"
            "2. El periodo o periodos de su disfrute se fijará de común acuerdo entre el "
            "empresario y el trabajador."
        ),
        titulo_ley="Título I",
        capitulo="Capítulo II",
        seccion="Sección 5.ª",
    )


@pytest.fixture
def articulo_largo() -> Articulo:
    # 3 apartados de 150 palabras: el artículo entero supera 400 tokens, cada apartado no.
    texto = "\n".join(_apartado(n, 150) for n in (1, 2, 3))
    return Articulo(numero="52", titulo="Extinción del contrato por causas objetivas", texto=texto)


# --- contar_tokens ---------------------------------------------------------------------------


def test_contar_tokens_vacio_es_cero() -> None:
    assert contar_tokens("") == 0


def test_contar_tokens_crece_con_el_texto() -> None:
    assert contar_tokens("uno dos tres cuatro") > contar_tokens("uno dos")


# --- encabezado ------------------------------------------------------------------------------


def test_encabezado_con_toda_la_jerarquia(articulo_corto: Articulo) -> None:
    assert encabezado(articulo_corto) == (
        "Estatuto de los Trabajadores · Título I · Capítulo II · Sección 5.ª · "
        "Art. 38. Vacaciones anuales"
    )


def test_encabezado_omite_niveles_vacios(articulo_largo: Articulo) -> None:
    assert encabezado(articulo_largo) == (
        "Estatuto de los Trabajadores · Art. 52. Extinción del contrato por causas objetivas"
    )


# --- trocear_articulo ------------------------------------------------------------------------


def test_articulo_corto_es_un_solo_chunk(articulo_corto: Articulo) -> None:
    chunks = trocear_articulo(articulo_corto)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.id == "38"
    assert chunk.articulo == "38"
    assert chunk.apartado is None
    assert chunk.padre is None
    assert "treinta días naturales" in chunk.texto


def test_articulo_largo_se_trocea_por_apartados(articulo_largo: Articulo) -> None:
    chunks = trocear_articulo(articulo_largo)

    assert [c.id for c in chunks] == ["52.1", "52.2", "52.3"]
    assert [c.apartado for c in chunks] == ["1", "2", "3"]
    assert all(c.padre == "52" and c.articulo == "52" for c in chunks)


def test_ningun_chunk_vacio_ni_por_encima_del_maximo(
    articulo_corto: Articulo, articulo_largo: Articulo
) -> None:
    for articulo in (articulo_corto, articulo_largo):
        for chunk in trocear_articulo(articulo):
            assert chunk.texto.strip()
            assert contar_tokens(chunk.texto) <= MAX_TOKENS


def test_texto_indexado_lleva_el_encabezado(articulo_corto: Articulo) -> None:
    chunk = trocear_articulo(articulo_corto)[0]

    assert chunk.texto_indexado.startswith(encabezado(articulo_corto))
    assert chunk.encabezado == encabezado(articulo_corto)


def test_parrafo_sin_numero_pertenece_al_apartado_anterior() -> None:
    # En el BOE un apartado ocupa a veces varios <p>: el art. 38.2 tiene un segundo párrafo
    # sin número ("En caso de desacuerdo…") que sigue siendo parte del 38.2.
    texto = "\n".join(
        [
            _apartado(1, 150),
            _apartado(2, 100),
            "En caso de desacuerdo entre las partes, la jurisdicción social fijará la fecha.",
            _apartado(3, 150),
        ]
    )
    articulo = Articulo(numero="38", titulo="Vacaciones anuales", texto=texto)

    chunks = trocear_articulo(articulo)

    assert [c.id for c in chunks] == ["38.1", "38.2", "38.3"]
    assert "En caso de desacuerdo" in chunks[1].texto


def test_apartado_demasiado_largo_se_divide_por_parrafos() -> None:
    # Un apartado de 3 párrafos de 150 palabras (~585 tokens) no cabe en 400: se parte en
    # trozos de párrafos completos, todos citables como "41.4".
    parrafo = " ".join(["trabajador"] * 150)
    texto = "\n".join([_apartado(1, 50), "4. " + parrafo, parrafo, parrafo, _apartado(5, 50)])
    articulo = Articulo(numero="41", titulo="Modificaciones sustanciales", texto=texto)

    chunks = [c for c in trocear_articulo(articulo) if c.apartado == "4"]

    assert len(chunks) > 1
    assert [c.id for c in chunks] == [f"41.4#{i}" for i in range(1, len(chunks) + 1)]
    assert all(contar_tokens(c.texto) <= MAX_TOKENS for c in chunks)


@pytest.mark.parametrize(
    ("numero", "esperado"),
    [
        ("38", "Art. 38"),
        ("20bis", "Art. 20 bis"),
        ("da1", "Disposición adicional 1.ª"),
        ("dt13", "Disposición transitoria 13.ª"),
    ],
)
def test_nombre_precepto_legible(numero: str, esperado: str) -> None:
    assert nombre_precepto(numero) == esperado
