"""Escribe results/recuperacion.md (tablas para el README) y results/recuperacion.json."""

import json
from dataclasses import asdict

from estatuto_rag.evals.golden import TIPOS
from estatuto_rag.evals.recuperacion import ResultadosRecuperacion
from estatuto_rag.ingest.pipeline import RAIZ

RESULTADOS = RAIZ / "results"


def tabla_markdown(resultados: ResultadosRecuperacion) -> str:
    lineas = [
        f"### Recuperación ({resultados.n_preguntas} preguntas con respuesta)",
        "",
        "| Estrategia | hit@1 | hit@5 | recall@5 | MRR@10 | p50 (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for nombre, m in resultados.global_.items():
        lineas.append(
            f"| {nombre} | {m['hit@1']:.2f} | {m['hit@5']:.2f} | {m['recall@5']:.2f} "
            f"| {m['mrr@10']:.2f} | {m['p50_ms']:.0f} |"
        )
    tipos = [t for t in TIPOS if any(t in pt for pt in resultados.por_tipo.values())]
    lineas += [
        "",
        "### hit@5 por tipo de pregunta",
        "",
        "| Estrategia | " + " | ".join(tipos) + " |",
        "|---|" + "---|" * len(tipos),
    ]
    for nombre, por_tipo in resultados.por_tipo.items():
        celdas = [f"{por_tipo[t]['hit@5']:.2f}" if t in por_tipo else "-" for t in tipos]
        lineas.append(f"| {nombre} | " + " | ".join(celdas) + " |")
    return "\n".join(lineas) + "\n"


def escribir_informe(resultados: ResultadosRecuperacion) -> None:
    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / "recuperacion.md").write_text(
        tabla_markdown(resultados), encoding="utf-8", newline="\n"
    )
    (RESULTADOS / "recuperacion.json").write_text(
        json.dumps(asdict(resultados), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
