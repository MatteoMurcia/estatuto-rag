"""`python -m estatuto_rag.retrieval "pregunta" [--modo vector|texto|hibrido] [-k 5]`."""

import argparse

from estatuto_rag.config import cargar_config
from estatuto_rag.embeddings import OllamaEmbedder
from estatuto_rag.retrieval.base import Recuperador
from estatuto_rag.retrieval.hibrido import RecuperadorHibrido
from estatuto_rag.retrieval.textual import RecuperadorTextual
from estatuto_rag.retrieval.vectorial import RecuperadorVectorial
from estatuto_rag.store.db import conectar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pregunta")
    parser.add_argument("--modo", choices=["vector", "texto", "hibrido"], default="hibrido")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    config = cargar_config()
    with conectar(config.database_url) as conn:
        vectorial = RecuperadorVectorial(
            conn, OllamaEmbedder(config.ollama_url, config.embed_model)
        )
        textual = RecuperadorTextual(conn)
        recuperadores: dict[str, Recuperador] = {
            "vector": vectorial,
            "texto": textual,
            "hibrido": RecuperadorHibrido([vectorial, textual]),
        }
        for r in recuperadores[args.modo].recuperar(args.pregunta, args.k):
            print(f"{r.score:.4f}  {r.id:<8} {r.encabezado.split(' · ')[-1][:70]}")


if __name__ == "__main__":
    main()
