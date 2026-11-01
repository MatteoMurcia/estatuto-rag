"""Evaluación de generación sobre el golden set (necesita Ollama; no corre en la CI).

`python -m estatuto_rag.evals.generacion [--modo hibrido hibrido+rerank] [--sin-juez]`

Métricas deterministas (sin juez):
- abstención correcta: en las preguntas sin respuesta, ¿dice que no lo encuentra?
- abstención indebida: en las que sí tienen respuesta, ¿se abstiene igualmente?
- citas inventadas: respuestas que citan un artículo que no estaba en el contexto
- cita correcta: respuestas que citan al menos uno de los artículos de referencia
Métricas con juez (phi4-mini): fidelidad al contexto y corrección frente a la referencia.
"""

import argparse
import json
import statistics
from dataclasses import asdict, dataclass

from estatuto_rag.evals.golden import PreguntaGolden
from estatuto_rag.evals.juez import Juez
from estatuto_rag.generation.llm import LLM
from estatuto_rag.generation.responder import responder
from estatuto_rag.retrieval.base import Recuperador


def cita_acierta(cita: str, referencia: str) -> bool:
    """'38.1' acierta '38' y '38.1'; '38' (artículo entero) también acierta '38.1'."""
    if cita.split(".")[0] != referencia.split(".")[0]:
        return False
    return "." not in cita or "." not in referencia or cita == referencia


@dataclass
class FilaGeneracion:
    id: str
    tipo: str
    pregunta: str
    respuesta: str
    citas: list[str]
    citas_invalidas: list[str]
    abstencion: bool
    fuentes: list[str]  # ids de los chunks del contexto, para poder recalcular citas sin rehacer
    cita_correcta: bool
    fiel: bool | None
    correcta: bool | None
    latencia_ms: float


def evaluar_pregunta(
    p: PreguntaGolden, recuperador: Recuperador, llm: LLM, juez: Juez | None
) -> FilaGeneracion:
    r = responder(p.pregunta, recuperador, llm)
    cita_correcta = any(cita_acierta(c, ref) for c in r.citas for ref in p.articulos_relevantes)
    fiel = correcta = None
    if juez is not None and not r.abstencion:
        fiel = juez.es_fiel(r.respuesta, r.fuentes)
    if juez is not None and p.tiene_respuesta:
        # Una abstención en una pregunta con respuesta es incorrecta sin preguntar al juez.
        correcta = (
            False
            if r.abstencion
            else juez.es_correcta(p.pregunta, p.respuesta_referencia, r.respuesta)
        )
    return FilaGeneracion(
        id=p.id,
        tipo=p.tipo,
        pregunta=p.pregunta,
        respuesta=r.respuesta,
        citas=r.citas,
        citas_invalidas=r.citas_invalidas,
        abstencion=r.abstencion,
        fuentes=[f.id for f in r.fuentes],
        cita_correcta=cita_correcta,
        fiel=fiel,
        correcta=correcta,
        latencia_ms=r.latencia_ms,
    )


def _tasa(valores: list[bool | None]) -> float | None:
    validos = [v for v in valores if v is not None]
    return sum(validos) / len(validos) if validos else None


def resumir(filas: list[FilaGeneracion]) -> dict[str, float | None]:
    con = [f for f in filas if f.tipo != "sin_respuesta"]
    sin = [f for f in filas if f.tipo == "sin_respuesta"]
    return {
        "abstencion_correcta": _tasa([f.abstencion for f in sin]),
        "abstencion_indebida": _tasa([f.abstencion for f in con]),
        "citas_inventadas": _tasa([bool(f.citas_invalidas) for f in filas]),
        "cita_correcta": _tasa([f.cita_correcta for f in con]),
        "fidelidad": _tasa([f.fiel for f in filas]),
        "correccion": _tasa([f.correcta for f in con]),
        "p50_ms": statistics.median(f.latencia_ms for f in filas) if filas else None,
    }


COLUMNAS = {
    "abstencion_correcta": "Abstención correcta ↑",
    "abstencion_indebida": "Abstención indebida ↓",
    "citas_inventadas": "Citas inventadas ↓",
    "cita_correcta": "Cita correcta ↑",
    "fidelidad": "Fidelidad (juez) ↑",
    "correccion": "Corrección (juez) ↑",
    "p50_ms": "p50 (s)",
}


def tabla_markdown(resumenes: dict[str, dict[str, float | None]]) -> str:
    def celda(clave: str, valor: float | None) -> str:
        if valor is None:
            return "-"
        return f"{valor / 1000:.1f}" if clave == "p50_ms" else f"{valor:.2f}"

    lineas = [
        "### Generación (40 preguntas)",
        "",
        "| Estrategia | " + " | ".join(COLUMNAS.values()) + " |",
        "|---|" + "---|" * len(COLUMNAS),
    ]
    for nombre, resumen in resumenes.items():
        lineas.append(f"| {nombre} | " + " | ".join(celda(c, resumen[c]) for c in COLUMNAS) + " |")
    return "\n".join(lineas) + "\n"


def main() -> None:
    from estatuto_rag.config import cargar_config
    from estatuto_rag.embeddings import OllamaEmbedder
    from estatuto_rag.evals.cache import EmbedderConCache, cargar_cache
    from estatuto_rag.evals.estrategias import NOMBRES, construir_estrategias
    from estatuto_rag.evals.golden import cargar_golden
    from estatuto_rag.evals.informe import RESULTADOS
    from estatuto_rag.generation.llm import OllamaLLM
    from estatuto_rag.store.db import conectar

    parser = argparse.ArgumentParser()
    parser.add_argument("--modo", nargs="+", choices=NOMBRES, default=["hibrido"])
    parser.add_argument("--sin-juez", action="store_true")
    args = parser.parse_args()

    config = cargar_config()
    embedder = EmbedderConCache(
        cargar_cache(), OllamaEmbedder(config.ollama_url, config.embed_model)
    )
    llm = OllamaLLM(config.ollama_url, config.llm_model)
    juez = None if args.sin_juez else Juez(OllamaLLM(config.ollama_url, config.judge_model))
    golden = cargar_golden()

    resumenes: dict[str, dict[str, float | None]] = {}
    detalle: dict[str, list[dict[str, object]]] = {}
    with conectar(config.database_url) as conn:
        estrategias = construir_estrategias(conn, embedder, "hibrido+rerank" in args.modo)
        for nombre in args.modo:
            filas = []
            for i, p in enumerate(golden, start=1):
                fila = evaluar_pregunta(p, estrategias[nombre], llm, juez)
                print(f"[{nombre}] {i}/{len(golden)} {p.id} {fila.latencia_ms / 1000:.1f}s")
                filas.append(fila)
            resumenes[nombre] = resumir(filas)
            detalle[nombre] = [asdict(f) for f in filas]

    RESULTADOS.mkdir(exist_ok=True)
    tabla = tabla_markdown(resumenes)
    (RESULTADOS / "generacion.md").write_text(tabla, encoding="utf-8", newline="\n")
    (RESULTADOS / "generacion.json").write_text(
        json.dumps({"resumen": resumenes, "detalle": detalle}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(tabla)


if __name__ == "__main__":
    main()
