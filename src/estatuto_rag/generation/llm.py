"""Cliente de LLM: Ollama en local (qwen2.5:7b por defecto)."""

from typing import Protocol, TypedDict

import httpx

# 5 chunks de hasta 400 palabras + prompt ≈ 3-4k tokens en el peor caso: 8k da margen.
NUM_CTX = 8192


class Mensaje(TypedDict):
    role: str
    content: str


class LLM(Protocol):
    def generar(self, mensajes: list[Mensaje]) -> str: ...


class OllamaLLM:
    def __init__(
        self, url: str, modelo: str, timeout: float = 120.0, num_ctx: int = NUM_CTX
    ) -> None:
        self._cliente = httpx.Client(base_url=url, timeout=timeout)
        self._modelo = modelo
        self._num_ctx = num_ctx

    def generar(self, mensajes: list[Mensaje]) -> str:
        respuesta = self._cliente.post(
            "/api/chat",
            json={
                "model": self._modelo,
                "messages": mensajes,
                "stream": False,
                # temperature 0: misma pregunta, misma respuesta.
                # num_ctx explícito: si el prompt no cabe en la ventana por defecto de Ollama,
                # Ollama recorta el principio SIN avisar y el modelo no ve parte del contexto.
                "options": {"temperature": 0, "num_ctx": self._num_ctx},
            },
        )
        respuesta.raise_for_status()
        contenido: str = respuesta.json()["message"]["content"]
        return contenido.strip()
