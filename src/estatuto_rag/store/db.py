"""Conexión a Postgres y esquema de la tabla `chunks`."""

import psycopg
from pgvector.psycopg import register_vector

ESQUEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id              text PRIMARY KEY,
    articulo        text NOT NULL,
    apartado        text,
    padre           text,
    encabezado      text NOT NULL,
    texto           text NOT NULL,
    texto_indexado  text NOT NULL,
    embedding       vector(1024) NOT NULL,
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('spanish', texto_indexado)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_tsv_gin
    ON chunks USING gin (tsv);
"""


def conectar(database_url: str) -> psycopg.Connection:
    """Abre una conexión, crea el esquema si no existe y registra el tipo `vector`."""
    conn = psycopg.connect(database_url, autocommit=True)
    conn.execute(ESQUEMA)
    register_vector(conn)
    return conn
