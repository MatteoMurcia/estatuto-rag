"""Tests de citas y del flujo responder() con un LLM falso."""

from estatuto_rag.generation.citas import (
    ABSTENCION,
    es_abstencion,
    extraer_citas,
    formatear_contexto,
    id_cita,
)
from estatuto_rag.generation.llm import Mensaje
from estatuto_rag.generation.responder import responder
from estatuto_rag.retrieval.base import Resultado


def _r(articulo: str, apartado: str | None, texto: str = "texto") -> Resultado:
    id_ = articulo if apartado is None else f"{articulo}.{apartado}"
    return Resultado(
        id=id_, articulo=articulo, apartado=apartado, encabezado="ET", texto=texto, score=1
    )


def test_id_cita_ignora_el_sufijo_de_parte() -> None:
    parte = Resultado(id="41.4#2", articulo="41", apartado="4", encabezado="", texto="", score=0)
    assert id_cita(parte) == "41.4"
    assert id_cita(_r("38", None)) == "38"


def test_formatear_contexto_usa_etiquetas_con_id() -> None:
    contexto = formatear_contexto([_r("38", "1", "treinta días")])
    assert contexto.startswith('<articulo id="38.1">')
    assert "treinta días" in contexto


def test_extraer_citas_unicas_y_en_orden() -> None:
    texto = "Son 30 días [art. 38.1]. Se pactan [art. 38.2, 38.1]. Ver también [art. 38.1]."
    assert extraer_citas(texto) == ["38.1", "38.2"]
    assert extraer_citas("Permiso parental [art. 48 bis.1].") == ["48bis.1"]


def test_es_abstencion() -> None:
    assert es_abstencion(ABSTENCION)
    assert not es_abstencion("Son 30 días naturales [art. 38.1].")


class RecuperadorFijo:
    def __init__(self, resultados: list[Resultado]) -> None:
        self.resultados = resultados

    def recuperar(self, pregunta: str, k: int) -> list[Resultado]:
        return self.resultados[:k]


class LLMFijo:
    def __init__(self, texto: str) -> None:
        self.texto = texto
        self.mensajes: list[Mensaje] = []

    def generar(self, mensajes: list[Mensaje]) -> str:
        self.mensajes = mensajes
        return self.texto


def test_responder_separa_citas_validas_e_inventadas() -> None:
    llm = LLMFijo("Son 30 días naturales [art. 38.1]. Y 2 meses de preaviso [art. 99].")

    r = responder("¿vacaciones?", RecuperadorFijo([_r("38", "1"), _r("38", "2")]), llm)

    assert r.citas == ["38.1"]
    assert r.citas_invalidas == ["99"]
    assert not r.abstencion
    assert '<articulo id="38.1">' in llm.mensajes[1]["content"]


def test_responder_acepta_citar_el_articulo_padre() -> None:
    r = responder("x", RecuperadorFijo([_r("38", "1")]), LLMFijo("Sí [art. 38]."))
    assert r.citas == ["38"]
    assert r.citas_invalidas == []


def test_responder_detecta_abstencion() -> None:
    r = responder("¿salario de un piloto?", RecuperadorFijo([_r("38", "1")]), LLMFijo(ABSTENCION))
    assert r.abstencion
    assert r.citas == []


def test_cita_valida_acepta_mas_detalle_que_el_chunk() -> None:
    from estatuto_rag.generation.citas import cita_valida

    entero, apartado = _r("38", None), _r("54", "2")
    assert cita_valida("38.1", [entero])  # el artículo 38 entero contiene el 38.1
    assert cita_valida("54.2.a", [apartado])
    assert cita_valida("54", [apartado])  # citar el artículo del apartado recuperado
    assert not cita_valida("54.3", [apartado])  # otro apartado: no estaba en el contexto
    assert not cita_valida("3", [entero])  # '3' no es prefijo de '38'
    assert not cita_valida("38", [_r("38bis", None)])
