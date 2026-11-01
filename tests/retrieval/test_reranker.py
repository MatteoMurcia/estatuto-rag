"""Tests del reranker con un puntuador falso (el modelo real no se usa en la CI)."""

from collections.abc import Sequence

from estatuto_rag.retrieval.base import Resultado
from estatuto_rag.retrieval.reranker import RecuperadorConReranker


def _r(id_: str, texto: str) -> Resultado:
    return Resultado(id=id_, articulo=id_, apartado=None, encabezado="", texto=texto, score=0.0)


class RecuperadorFijo:
    def __init__(self, resultados: list[Resultado]) -> None:
        self.resultados = resultados
        self.k_pedido: int | None = None

    def recuperar(self, pregunta: str, k: int) -> list[Resultado]:
        self.k_pedido = k
        return self.resultados[:k]


class PuntuadorPorPalabra:
    """Puntúa con 1.0 los textos que contienen la palabra clave."""

    def __init__(self, palabra: str) -> None:
        self.palabra = palabra

    def puntuar(self, pregunta: str, textos: Sequence[str]) -> list[float]:
        return [1.0 if self.palabra in t else 0.0 for t in textos]


def test_reordena_por_la_puntuacion_del_cross_encoder() -> None:
    base = RecuperadorFijo([_r("a", "jornada"), _r("b", "vacaciones"), _r("c", "salario")])
    reranker = RecuperadorConReranker(base, PuntuadorPorPalabra("vacaciones"), candidatos=3)

    resultados = reranker.recuperar("vacaciones", 2)

    assert [r.id for r in resultados] == ["b", "a"]
    assert resultados[0].score == 1.0


def test_pide_mas_candidatos_de_los_que_devuelve() -> None:
    base = RecuperadorFijo([_r(str(i), "x") for i in range(50)])
    reranker = RecuperadorConReranker(base, PuntuadorPorPalabra("x"), candidatos=30)

    resultados = reranker.recuperar("x", 5)

    assert base.k_pedido == 30
    assert len(resultados) == 5


def test_sin_candidatos_no_llama_al_modelo() -> None:
    reranker = RecuperadorConReranker(RecuperadorFijo([]), PuntuadorPorPalabra("x"))

    assert reranker.recuperar("x", 5) == []
