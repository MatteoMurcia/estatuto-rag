"""Tests de la evaluación de recuperación y del informe, con recuperadores falsos."""

from estatuto_rag.evals.golden import PreguntaGolden
from estatuto_rag.evals.informe import tabla_markdown
from estatuto_rag.evals.recuperacion import evaluar_recuperacion
from estatuto_rag.retrieval.base import Resultado


def _p(id_: str, refs: list[str], tipo: str = "factual") -> PreguntaGolden:
    return PreguntaGolden(id_, f"pregunta {id_}", refs, "", tipo, True)


def _r(articulo: str) -> Resultado:
    return Resultado(articulo, articulo, None, "", "", 0.0)


class _Fijo:
    """Devuelve siempre la misma lista de artículos."""

    def __init__(self, articulos: list[str]) -> None:
        self.articulos = articulos

    def recuperar(self, pregunta: str, k: int) -> list[Resultado]:
        return [_r(a) for a in self.articulos[:k]]


GOLDEN = [
    _p("f1", ["38"]),
    _p("p1", ["54"], "parafraseada"),
    _p("u1", [], "sin_respuesta"),
]


def test_metricas_globales_y_por_tipo() -> None:
    res = evaluar_recuperacion(GOLDEN, {"a": _Fijo(["38", "1", "2", "3", "4", "54"])})
    assert res.n_preguntas == 2  # la sin respuesta no cuenta en recuperación
    m = res.global_["a"]
    assert m["hit@1"] == 0.5
    assert m["hit@5"] == 0.5  # 54 está en la posición 6
    assert m["mrr@10"] == (1 + 1 / 6) / 2
    assert res.por_tipo["a"]["factual"]["hit@5"] == 1.0
    assert res.por_tipo["a"]["parafraseada"]["hit@5"] == 0.0
    assert res.fallos == {"a": ["p1"]}


def test_tabla_markdown_tiene_una_fila_por_estrategia() -> None:
    res = evaluar_recuperacion(GOLDEN, {"a": _Fijo(["38"]), "b": _Fijo(["54"])})
    tabla = tabla_markdown(res)
    assert "| a | 0.50 | 0.50 |" in tabla
    assert "| b | 0.50 | 0.50 |" in tabla
    assert "sin_respuesta" not in tabla
