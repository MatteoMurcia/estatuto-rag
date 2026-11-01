"""Configuración leída de variables de entorno, con valores por defecto para desarrollo local."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    database_url: str
    ollama_url: str
    embed_model: str
    llm_model: str
    judge_model: str


def cargar_config() -> Config:
    """Se llama al usarla, no al importar el módulo: así importar nunca falla por el entorno."""
    return Config(
        database_url=os.environ.get(
            "DATABASE_URL", "postgresql://estatuto:estatuto@localhost:5432/estatuto"
        ),
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        embed_model=os.environ.get("EMBED_MODEL", "bge-m3"),
        llm_model=os.environ.get("LLM_MODEL", "qwen2.5:7b"),
        judge_model=os.environ.get("JUDGE_MODEL", "phi4-mini"),
    )
