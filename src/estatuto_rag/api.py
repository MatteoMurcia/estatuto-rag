"""API HTTP: `uv run uvicorn estatuto_rag.api:app --reload`."""

from collections.abc import Iterator
from typing import Annotated, Literal

import psycopg
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from estatuto_rag.config import cargar_config
from estatuto_rag.embeddings import OllamaEmbedder
from estatuto_rag.generation.llm import LLM, OllamaLLM
from estatuto_rag.generation.responder import responder
from estatuto_rag.retrieval.base import Recuperador
from estatuto_rag.retrieval.hibrido import RecuperadorHibrido
from estatuto_rag.retrieval.textual import RecuperadorTextual
from estatuto_rag.retrieval.vectorial import RecuperadorVectorial
from estatuto_rag.store.db import conectar

app = FastAPI(title="estatuto-rag", description="No es asesoramiento jurídico.")

Modo = Literal["vector", "texto", "hibrido"]


class Pregunta(BaseModel):
    pregunta: str = Field(min_length=3, max_length=500)
    modo: Modo = "hibrido"
    k: int = Field(default=5, ge=1, le=20)


class Fuente(BaseModel):
    id: str
    encabezado: str
    texto: str
    score: float


class RespuestaAPI(BaseModel):
    respuesta: str
    citas: list[str]
    citas_invalidas: list[str]
    abstencion: bool
    fuentes: list[Fuente]
    latencia_ms: float


class Servicio:
    """Agrupa conexión, recuperadores y LLM. En los tests se sustituye por uno falso."""

    def __init__(self, conn: psycopg.Connection) -> None:
        config = cargar_config()
        vectorial = RecuperadorVectorial(
            conn, OllamaEmbedder(config.ollama_url, config.embed_model)
        )
        textual = RecuperadorTextual(conn)
        self.recuperadores: dict[str, Recuperador] = {
            "vector": vectorial,
            "texto": textual,
            "hibrido": RecuperadorHibrido([vectorial, textual]),
        }
        self.llm: LLM = OllamaLLM(config.ollama_url, config.llm_model)


def obtener_servicio() -> Iterator[Servicio]:
    with conectar(cargar_config().database_url) as conn:
        yield Servicio(conn)


@app.post("/ask")
def ask(datos: Pregunta, servicio: Annotated[Servicio, Depends(obtener_servicio)]) -> RespuestaAPI:
    r = responder(datos.pregunta, servicio.recuperadores[datos.modo], servicio.llm, datos.k)
    return RespuestaAPI(
        respuesta=r.respuesta,
        citas=r.citas,
        citas_invalidas=r.citas_invalidas,
        abstencion=r.abstencion,
        fuentes=[
            Fuente(id=f.id, encabezado=f.encabezado, texto=f.texto, score=f.score)
            for f in r.fuentes
        ],
        latencia_ms=round(r.latencia_ms, 1),
    )
