"""Búsqueda léxica: full-text search de Postgres con el diccionario español."""

import psycopg

from estatuto_rag.retrieval.base import COLUMNAS, Resultado

# plainto_tsquery une los términos con AND (&): una pregunta natural casi nunca contiene
# TODAS sus palabras en un mismo artículo. Los cambiamos por OR (|) y dejamos que
# ts_rank_cd premie a los chunks que contienen más términos y más juntos.
SQL = f"""
WITH q AS (
    SELECT to_tsquery('spanish', replace(plainto_tsquery('spanish', %(pregunta)s)::text, '&', '|'))
        AS consulta
)
SELECT {COLUMNAS}, ts_rank_cd(tsv, q.consulta) AS score
FROM chunks, q
WHERE tsv @@ q.consulta
ORDER BY score DESC, id
LIMIT %(k)s
"""


class RecuperadorTextual:
    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def recuperar(self, pregunta: str, k: int) -> list[Resultado]:
        filas = self._conn.execute(SQL, {"pregunta": pregunta, "k": k}).fetchall()
        return [Resultado(*fila) for fila in filas]
