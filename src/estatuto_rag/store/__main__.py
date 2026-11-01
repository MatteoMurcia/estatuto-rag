"""`python -m estatuto_rag.store`: carga data/chunks.jsonl en Postgres con embeddings de bge-m3."""

import time

from estatuto_rag.config import cargar_config
from estatuto_rag.embeddings import OllamaEmbedder
from estatuto_rag.ingest.pipeline import CHUNKS_JSONL, leer_jsonl
from estatuto_rag.store.db import conectar
from estatuto_rag.store.indexar import indexar


def main() -> None:
    config = cargar_config()
    chunks = leer_jsonl(CHUNKS_JSONL)
    embedder = OllamaEmbedder(config.ollama_url, config.embed_model)
    inicio = time.perf_counter()
    with conectar(config.database_url) as conn:
        n = indexar(conn, chunks, embedder)
    print(f"{n} chunks indexados en {time.perf_counter() - inicio:.1f} s")


if __name__ == "__main__":
    main()
