"""`python -m estatuto_rag.evals`: compara estrategias de recuperación sobre el golden set.

Opciones:
  --rerank           incluye híbrido + reranker (necesita `uv sync --extra rerank`)
  --solo-cache       usa solo embeddings precalculados (lo que hace la CI, sin Ollama)
  --min-hit5 0.70    sale con error si el hit@5 del híbrido baja de ese umbral (gate de CI)
  --reescritura      incluye híbrido + reescritura de la consulta con el LLM (necesita Ollama)
  --cache RUTA       caché de embeddings a usar (por defecto data/embeddings/bge-m3.jsonl.gz)
"""

import argparse
import sys
from pathlib import Path

from estatuto_rag.config import cargar_config
from estatuto_rag.embeddings import Embedder, OllamaEmbedder
from estatuto_rag.evals.cache import CACHE, EmbedderConCache, cargar_cache
from estatuto_rag.evals.estrategias import construir_estrategias
from estatuto_rag.evals.golden import cargar_golden
from estatuto_rag.evals.informe import RESULTADOS, escribir_informe
from estatuto_rag.evals.recuperacion import evaluar_recuperacion
from estatuto_rag.ingest.pipeline import CHUNKS_JSONL, leer_jsonl
from estatuto_rag.store.db import conectar
from estatuto_rag.store.indexar import indexar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--solo-cache", action="store_true")
    parser.add_argument("--min-hit5", type=float, default=None)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--reescritura", action="store_true")
    args = parser.parse_args()
    if args.reescritura and args.solo_cache:
        parser.error("--reescritura genera consultas nuevas: sus embeddings no están en la caché")

    config = cargar_config()
    respaldo: Embedder | None = None
    if not args.solo_cache:
        respaldo = OllamaEmbedder(config.ollama_url, config.embed_model)
    embedder = EmbedderConCache(cargar_cache(args.cache), respaldo)

    with conectar(config.database_url) as conn:
        indexar(conn, leer_jsonl(CHUNKS_JSONL), embedder)
        estrategias = construir_estrategias(conn, embedder, args.rerank)
        if args.reescritura:
            from estatuto_rag.generation.llm import OllamaLLM
            from estatuto_rag.retrieval.reescritura import RecuperadorConReescritura, Reescritor

            reescritor = Reescritor(OllamaLLM(config.ollama_url, config.llm_model))
            estrategias["hibrido+reescritura"] = RecuperadorConReescritura(
                estrategias["hibrido"], reescritor
            )
        resultados = evaluar_recuperacion(cargar_golden(), estrategias)

    escribir_informe(resultados)
    print((RESULTADOS / "recuperacion.md").read_text(encoding="utf-8"))

    if args.min_hit5 is not None:
        hit5 = resultados.global_["hibrido"]["hit@5"]
        if hit5 < args.min_hit5:
            print(f"FALLO: hit@5 del híbrido = {hit5:.3f} < umbral {args.min_hit5:.3f}")
            sys.exit(1)
        print(f"OK: hit@5 del híbrido = {hit5:.3f} >= umbral {args.min_hit5:.3f}")


if __name__ == "__main__":
    main()
