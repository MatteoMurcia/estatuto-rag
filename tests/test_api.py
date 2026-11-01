"""Tests de la API con el servicio sustituido por uno falso (sin BD ni Ollama)."""

from fastapi.testclient import TestClient

from estatuto_rag.api import Servicio, app, obtener_servicio
from estatuto_rag.generation.llm import Mensaje
from estatuto_rag.retrieval.base import Resultado

FUENTE = Resultado(
    id="38", articulo="38", apartado=None, encabezado="ET · Art. 38", texto="treinta", score=0.9
)


class RecuperadorFijo:
    def recuperar(self, pregunta: str, k: int) -> list[Resultado]:
        return [FUENTE]


class LLMFijo:
    def generar(self, mensajes: list[Mensaje]) -> str:
        return "Son 30 días naturales [art. 38]."


class ServicioFalso(Servicio):
    def __init__(self) -> None:
        self.recuperadores = {
            "vector": RecuperadorFijo(),
            "texto": RecuperadorFijo(),
            "hibrido": RecuperadorFijo(),
        }
        self.llm = LLMFijo()


app.dependency_overrides[obtener_servicio] = ServicioFalso
cliente = TestClient(app)


def test_ask_devuelve_respuesta_con_citas() -> None:
    respuesta = cliente.post("/ask", json={"pregunta": "¿Cuántos días de vacaciones?"})

    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["citas"] == ["38"]
    assert datos["abstencion"] is False
    assert datos["fuentes"][0]["id"] == "38"
    assert datos["latencia_ms"] >= 0


def test_ask_valida_la_entrada() -> None:
    assert cliente.post("/ask", json={"pregunta": ""}).status_code == 422
    assert cliente.post("/ask", json={"pregunta": "x" * 501}).status_code == 422
    assert (
        cliente.post("/ask", json={"pregunta": "hola que tal", "modo": "otro"}).status_code == 422
    )
