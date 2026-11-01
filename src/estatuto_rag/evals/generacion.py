"""Evaluación de generación sobre el golden set (necesita Ollama; no corre en la CI).

`python -m estatuto_rag.evals.generacion [--modo hibrido hibrido+rerank] [--sin-juez]`

Métricas deterministas (sin juez):
- abstención correcta: en las preguntas sin respuesta, ¿dice que no lo encuentra?
- abstención indebida: en las que sí tienen respuesta, ¿se abstiene igualmente?
- citas inventadas: respuestas que citan un artículo que no estaba en el contexto
- cita correcta: respuestas que citan al menos uno de los artículos de referencia
- cifras fieles: respuestas cuyos números aparecen todos en el contexto (evals/numeros.py)
Métricas con juez (phi4-mini): fidelidad al contexto y corrección frente a la referencia.

`--recalcular` rehace las métricas deterministas desde results/generacion.json sin llamar a
Ollama (útil al cambiar una métrica: las respuestas guardadas no cambian).
"""

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from estatuto_rag.evals.golden import PreguntaGolden
from estatuto_rag.evals.juez import Juez
from estatuto_rag.evals.metricas import hit_at_k
from estatuto_rag.evals.numeros import numeros_no_respaldados
from estatuto_rag.generation.citas import cita_valida, extraer_citas
from estatuto_rag.generation.llm import LLM
from estatuto_rag.generation.responder import Respuesta, responder
from estatuto_rag.ingest.models import Chunk
from estatuto_rag.retrieval.base import Recuperador, Resultado


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
    consultas: list[str]  # búsquedas hechas (el agente hace varias)
    contexto_acierta: bool  # ¿llegó al contexto algún artículo de referencia? (hit@k)
    fuentes: list[str]  # ids de los chunks del contexto, para poder recalcular citas sin rehacer
    cita_correcta: bool
    cifras_inventadas: list[int]
    fiel: bool | None
    correcta: bool | None
    latencia_ms: float


def _cifras_inventadas(respuesta: str, fuentes: list[Resultado], abstencion: bool) -> list[int]:
    if abstencion:
        return []
    contexto = "\n".join(f.texto for f in fuentes)
    return sorted(numeros_no_respaldados(respuesta, contexto))


Responder = Callable[[str], Respuesta]  # RAG simple o agente: pregunta → respuesta


def rag_simple(recuperador: Recuperador, llm: LLM) -> Responder:
    return lambda pregunta: responder(pregunta, recuperador, llm)


def evaluar_pregunta(
    p: PreguntaGolden, responder_fn: Responder, juez: Juez | None
) -> FilaGeneracion:
    r = responder_fn(p.pregunta)
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
        consultas=r.consultas,
        contexto_acierta=bool(hit_at_k(r.fuentes, p.articulos_relevantes, len(r.fuentes))),
        fuentes=[f.id for f in r.fuentes],
        cita_correcta=cita_correcta,
        cifras_inventadas=_cifras_inventadas(r.respuesta, r.fuentes, r.abstencion),
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
        "contexto_acierta": _tasa([f.contexto_acierta for f in con]),
        "abstencion_correcta": _tasa([f.abstencion for f in sin]),
        "abstencion_indebida": _tasa([f.abstencion for f in con]),
        "citas_inventadas": _tasa([bool(f.citas_invalidas) for f in filas]),
        "cita_correcta": _tasa([f.cita_correcta for f in con]),
        "cifras_fieles": _tasa([not f.cifras_inventadas for f in filas if not f.abstencion]),
        "fidelidad": _tasa([f.fiel for f in filas]),
        "correccion": _tasa([f.correcta for f in con]),
        "p50_ms": statistics.median(f.latencia_ms for f in filas) if filas else None,
    }


COLUMNAS = {
    "contexto_acierta": "Artículo en contexto ↑",
    "abstencion_correcta": "Abstención correcta ↑",
    "abstencion_indebida": "Abstención indebida ↓",
    "citas_inventadas": "Citas inventadas ↓",
    "cita_correcta": "Cita correcta ↑",
    "cifras_fieles": "Cifras fieles ↑",
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


def recalcular(
    detalle: dict[str, list[dict[str, Any]]], golden: list[PreguntaGolden], chunks: list[Chunk]
) -> dict[str, list[FilaGeneracion]]:
    """Rehace citas y cifras de respuestas ya guardadas, a partir de los ids de sus fuentes."""
    por_id = {c.id: c for c in chunks}
    refs = {p.id: p.articulos_relevantes for p in golden}
    salida: dict[str, list[FilaGeneracion]] = {}
    for nombre, filas in detalle.items():
        nuevas = []
        for d in filas:
            fuentes = [_resultado(por_id[i]) for i in d["fuentes"]]
            citas = extraer_citas(d["respuesta"])
            validas = [c for c in citas if cita_valida(c, fuentes)]
            d = d | {
                "citas": validas,
                "citas_invalidas": [c for c in citas if c not in validas],
                "cita_correcta": any(cita_acierta(c, r) for c in validas for r in refs[d["id"]]),
                "cifras_inventadas": _cifras_inventadas(d["respuesta"], fuentes, d["abstencion"]),
                "consultas": d.get("consultas") or [d["pregunta"]],
                "contexto_acierta": bool(hit_at_k(fuentes, refs[d["id"]], len(fuentes))),
            }
            nuevas.append(FilaGeneracion(**d))
        salida[nombre] = nuevas
    return salida


def _resultado(c: Chunk) -> Resultado:
    return Resultado(c.id, c.articulo, c.apartado, c.encabezado, c.texto, 0.0)


def guardar(filas_por_modo: dict[str, list[FilaGeneracion]]) -> str:
    from estatuto_rag.evals.informe import RESULTADOS

    resumenes = {nombre: resumir(filas) for nombre, filas in filas_por_modo.items()}
    detalle = {nombre: [asdict(f) for f in filas] for nombre, filas in filas_por_modo.items()}
    RESULTADOS.mkdir(exist_ok=True)
    tabla = tabla_markdown(resumenes)
    (RESULTADOS / "generacion.md").write_text(tabla, encoding="utf-8", newline="\n")
    (RESULTADOS / "generacion.json").write_text(
        json.dumps({"resumen": resumenes, "detalle": detalle}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return tabla


def main() -> None:
    from estatuto_rag.agente.grafo import Agente
    from estatuto_rag.config import cargar_config
    from estatuto_rag.embeddings import OllamaEmbedder
    from estatuto_rag.evals.cache import EmbedderConCache, cargar_cache
    from estatuto_rag.evals.estrategias import NOMBRES, construir_estrategias
    from estatuto_rag.evals.golden import cargar_golden
    from estatuto_rag.evals.informe import RESULTADOS
    from estatuto_rag.generation.llm import OllamaLLM
    from estatuto_rag.ingest.pipeline import CHUNKS_JSONL, leer_jsonl
    from estatuto_rag.retrieval.reescritura import Reescritor
    from estatuto_rag.store.db import conectar

    parser = argparse.ArgumentParser()
    parser.add_argument("--modo", nargs="+", choices=[*NOMBRES, "agente"], default=["hibrido"])
    parser.add_argument("--sin-juez", action="store_true")
    parser.add_argument("--recalcular", action="store_true")
    args = parser.parse_args()
    golden = cargar_golden()

    if args.recalcular:
        guardado = json.loads((RESULTADOS / "generacion.json").read_text(encoding="utf-8"))
        print(guardar(recalcular(guardado["detalle"], golden, leer_jsonl(CHUNKS_JSONL))))
        return

    config = cargar_config()
    embedder = EmbedderConCache(
        cargar_cache(), OllamaEmbedder(config.ollama_url, config.embed_model)
    )
    llm = OllamaLLM(config.ollama_url, config.llm_model)
    juez = None if args.sin_juez else Juez(OllamaLLM(config.ollama_url, config.judge_model))

    filas_por_modo: dict[str, list[FilaGeneracion]] = {}
    with conectar(config.database_url) as conn:
        estrategias = construir_estrategias(conn, embedder, "hibrido+rerank" in args.modo)
        for nombre in args.modo:
            filas = []
            if nombre == "agente":
                # El agente usa el híbrido como buscador base: mide solo reescritura + reintento.
                agente = Agente(estrategias["hibrido"], llm, Reescritor(llm))
                responder_fn: Responder = agente.responder
            else:
                responder_fn = rag_simple(estrategias[nombre], llm)
            for i, p in enumerate(golden, start=1):
                fila = evaluar_pregunta(p, responder_fn, juez)
                print(f"[{nombre}] {i}/{len(golden)} {p.id} {fila.latencia_ms / 1000:.1f}s")
                filas.append(fila)
            filas_por_modo[nombre] = filas
    print(guardar(filas_por_modo))


if __name__ == "__main__":
    main()
