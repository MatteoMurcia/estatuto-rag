"""Evaluación de recuperación: ejecuta cada estrategia sobre cada pregunta y agrega métricas."""

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field

from estatuto_rag.evals.golden import PreguntaGolden
from estatuto_rag.evals.metricas import hit_at_k, recall_at_k, reciprocal_rank
from estatuto_rag.retrieval.base import Recuperador

K_MAX = 10

Metricas = dict[str, float]


@dataclass
class ResultadosRecuperacion:
    global_: dict[str, Metricas] = field(default_factory=dict)
    por_tipo: dict[str, dict[str, Metricas]] = field(default_factory=dict)
    n_preguntas: int = 0
    fallos: dict[str, list[str]] = field(default_factory=dict)  # preguntas sin acierto en top 5


def _media(valores: list[float]) -> float:
    return statistics.fmean(valores) if valores else 0.0


def evaluar_recuperacion(
    golden: list[PreguntaGolden], estrategias: dict[str, Recuperador]
) -> ResultadosRecuperacion:
    # Las preguntas sin respuesta no tienen artículo correcto: se evalúan en generación.
    preguntas = [p for p in golden if p.tiene_respuesta]
    salida = ResultadosRecuperacion(n_preguntas=len(preguntas))

    for nombre, recuperador in estrategias.items():
        filas: list[tuple[str, Metricas]] = []
        latencias: list[float] = []
        for p in preguntas:
            inicio = time.perf_counter()
            resultados = recuperador.recuperar(p.pregunta, K_MAX)
            latencias.append((time.perf_counter() - inicio) * 1000)
            ref = p.articulos_relevantes
            filas.append(
                (
                    p.tipo,
                    {
                        "hit@1": hit_at_k(resultados, ref, 1),
                        "hit@5": hit_at_k(resultados, ref, 5),
                        "recall@5": recall_at_k(resultados, ref, 5),
                        "mrr@10": reciprocal_rank(resultados, ref),
                    },
                )
            )
            if not hit_at_k(resultados, ref, 5):
                salida.fallos.setdefault(nombre, []).append(p.id)

        claves = ["hit@1", "hit@5", "recall@5", "mrr@10"]
        salida.global_[nombre] = {c: _media([m[c] for _, m in filas]) for c in claves}
        salida.global_[nombre]["p50_ms"] = statistics.median(latencias) if latencias else 0.0

        agrupadas: dict[str, list[Metricas]] = defaultdict(list)
        for tipo, m in filas:
            agrupadas[tipo].append(m)
        salida.por_tipo[nombre] = {
            tipo: {c: _media([m[c] for m in ms]) for c in claves} for tipo, ms in agrupadas.items()
        }
    return salida
