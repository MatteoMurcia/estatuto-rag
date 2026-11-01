"""`python -m estatuto_rag.ingest`: genera data/chunks.jsonl a partir del XML del BOE."""

from estatuto_rag.ingest.pipeline import CHUNKS_JSONL, XML_BOE, construir_chunks, guardar_jsonl


def main() -> None:
    chunks = construir_chunks(XML_BOE.read_text(encoding="utf-8"))
    guardar_jsonl(chunks, CHUNKS_JSONL)
    articulos = len({c.articulo for c in chunks})
    print(f"{len(chunks)} chunks de {articulos} preceptos -> {CHUNKS_JSONL}")


if __name__ == "__main__":
    main()
