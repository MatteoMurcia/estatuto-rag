"""Embeddings precalculados para que la CI evalúe con bge-m3 real sin tener Ollama.

`python -m estatuto_rag.evals.cache` calcula (con Ollama) los vectores de todos los chunks y de
todas las preguntas del golden set y los guarda comprimidos en data/embeddings/.
"""

import gzip
import hashlib
import json
from pathlib import Path

from estatuto_rag.embeddings import Embedder
from estatuto_rag.ingest.pipeline import RAIZ

CACHE = RAIZ / "data" / "embeddings" / "bge-m3.jsonl.gz"


def clave(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def cargar_cache(ruta: Path = CACHE) -> dict[str, list[float]]:
    if not ruta.exists():
        return {}
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        filas = (json.loads(linea) for linea in f)
        return {fila["clave"]: fila["vector"] for fila in filas}


def guardar_cache(vectores: dict[str, list[float]], ruta: Path = CACHE) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 y sin nombre: el .gz sale idéntico byte a byte si no cambian los vectores.
    with (
        open(ruta, "wb") as bruto,
        gzip.GzipFile(filename="", fileobj=bruto, mode="wb", mtime=0) as comprimido,
    ):
        for k in sorted(vectores):
            fila = {"clave": k, "vector": [round(x, 6) for x in vectores[k]]}
            comprimido.write((json.dumps(fila) + "\n").encode("utf-8"))


class EmbedderConCache:
    """Busca cada texto en la caché; si falta, lo pide al embedder de respaldo (si lo hay)."""

    def __init__(self, vectores: dict[str, list[float]], respaldo: Embedder | None = None) -> None:
        self.vectores = vectores
        self._respaldo = respaldo

    def embed(self, textos: list[str]) -> list[list[float]]:
        faltan = [t for t in textos if clave(t) not in self.vectores]
        if faltan:
            if self._respaldo is None:
                raise KeyError(
                    f"{len(faltan)} textos sin embedding en caché (p. ej. {faltan[0][:60]!r}). "
                    "Regenera la caché con: python -m estatuto_rag.evals.cache"
                )
            for texto, vector in zip(faltan, self._respaldo.embed(faltan), strict=True):
                self.vectores[clave(texto)] = vector
        return [self.vectores[clave(t)] for t in textos]


def main() -> None:
    from estatuto_rag.config import cargar_config
    from estatuto_rag.embeddings import OllamaEmbedder
    from estatuto_rag.evals.golden import cargar_golden
    from estatuto_rag.ingest.pipeline import CHUNKS_JSONL, leer_jsonl

    config = cargar_config()
    textos = [c.texto_indexado for c in leer_jsonl(CHUNKS_JSONL)]
    textos += [p.pregunta for p in cargar_golden()]
    # Caché nueva desde cero: así no quedan vectores de textos que ya no existen.
    embedder = EmbedderConCache({}, OllamaEmbedder(config.ollama_url, config.embed_model))
    for inicio in range(0, len(textos), 32):
        embedder.embed(textos[inicio : inicio + 32])
    guardar_cache(embedder.vectores)
    print(f"{len(embedder.vectores)} embeddings guardados en {CACHE}")


if __name__ == "__main__":
    main()
