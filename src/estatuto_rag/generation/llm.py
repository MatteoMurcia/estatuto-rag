"""Cliente de LLM: Ollama en local (qwen2.5:7b por defecto)."""

from typing import Protocol, TypedDict

import httpx


class Mensaje(TypedDict):
    role: str
    content: str


class LLM(Protocol):
    def generar(self, mensajes: list[Mensaje]) -> str: ...


class OllamaLLM:
    def __init__(self, url: str, modelo: str, timeout: float = 120.0) -> None:
        self._cliente = httpx.Client(base_url=url, timeout=timeout)
        self._modelo = modelo

    def generar(self, mensajes: list[Mensaje]) -> str:
        respuesta = self._cliente.post(
            "/api/chat",
            json={
                "model": self._modelo,
                "messages": mensajes,
                "stream": False,
                "options": {"temperature": 0},  # determinista: misma pregunta, misma respuesta
            },
        )
        respuesta.raise_for_status()
        contenido: str = respuesta.json()["message"]["content"]
        return contenido.strip()
