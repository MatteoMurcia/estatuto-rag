"""Genera los gráficos SVG del README (docs/img/) sin dependencias externas.

Los valores son los de las tablas del README (results/ y la ejecución del reranker).
Cada SVG trae su propio modo claro y oscuro, así que se ve bien en GitHub con
cualquiera de los dos temas.

    python scripts/graficos.py
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

SALIDA = Path(__file__).resolve().parent.parent / "docs" / "img"

ESTILO = """
<style>
  .fondo { fill: #fcfcfb; }
  .t1 { fill: #0b0b0b; font: 600 15px system-ui, -apple-system, 'Segoe UI', sans-serif; }
  .t2 { fill: #52514e; font: 12px system-ui, -apple-system, 'Segoe UI', sans-serif; }
  .val { fill: #0b0b0b; font: 600 11px system-ui, -apple-system, 'Segoe UI', sans-serif; }
  .rejilla { stroke: #e4e3df; stroke-width: 1; }
  .eje { stroke: #b5b4ad; stroke-width: 1; }
  .s1 { fill: #2a78d6; } .s2 { fill: #eb6834; } .s3 { fill: #1baf7a; }
  .trama { stroke: #eb6834; stroke-width: 1.5; }
  .borde { stroke: #eb6834; stroke-width: 1; }
  @media (prefers-color-scheme: dark) {
    .fondo { fill: #1a1a19; }
    .t1, .val { fill: #f0efec; }
    .t2 { fill: #c3c2b7; }
    .rejilla { stroke: #2e2e2c; }
    .eje { stroke: #5a5955; }
    .s1 { fill: #3987e5; } .s2 { fill: #d95926; } .s3 { fill: #199e70; }
    .trama { stroke: #d95926; }
    .borde { stroke: #d95926; }
  }
</style>
<defs>
  <pattern id="estimado" width="6" height="6" patternUnits="userSpaceOnUse"
           patternTransform="rotate(45)">
    <line x1="0" y1="0" x2="0" y2="6" class="trama"/>
  </pattern>
</defs>
"""


def _texto(x: float, y: float, s: str, clase: str, ancla: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{clase}" text-anchor="{ancla}">{escape(s)}</text>'


def _barra_v(x: float, base: float, ancho: float, alto: float, clase: str) -> str:
    """Barra vertical con el extremo superior redondeado (4 px) y la base recta."""
    r = min(4.0, alto, ancho / 2)
    top = base - alto
    return (
        f'<path class="{clase}" d="M{x:.1f},{base:.1f} V{top + r:.1f} '
        f"Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} H{x + ancho - r:.1f} "
        f'Q{x + ancho:.1f},{top:.1f} {x + ancho:.1f},{top + r:.1f} V{base:.1f} Z"/>'
    )


def _barra_h(x: float, y: float, largo: float, grosor: float, relleno: str) -> str:
    """Barra horizontal con el extremo derecho redondeado (4 px)."""
    r = min(4.0, largo, grosor / 2)
    fin = x + largo
    return (
        f'<path {relleno} d="M{x:.1f},{y:.1f} H{fin - r:.1f} '
        f"Q{fin:.1f},{y:.1f} {fin:.1f},{y + r:.1f} V{y + grosor - r:.1f} "
        f'Q{fin:.1f},{y + grosor:.1f} {fin - r:.1f},{y + grosor:.1f} H{x:.1f} Z"/>'
    )


def _leyenda(x: float, y: float, series: list[tuple[str, str]]) -> list[str]:
    partes = []
    for clase, nombre in series:
        partes.append(f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2" {clase}/>')
        partes.append(_texto(x + 15, y, nombre, "t2"))
        x += 22 + 7 * len(nombre)
    return partes


def _svg(ancho: int, alto: int, titulo: str, cuerpo: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" '
        f'viewBox="0 0 {ancho} {alto}" role="img" aria-label="{escape(titulo)}">'
        f"<title>{escape(titulo)}</title>{ESTILO}"
        f'<rect class="fondo" width="{ancho}" height="{alto}" rx="8"/>'
        + "".join(cuerpo)
        + "</svg>\n"
    )


def barras_agrupadas(
    titulo: str,
    subtitulo: str,
    grupos: list[str],
    series: list[tuple[str, str, list[float]]],
    formato: str = "{:.2f}",
) -> str:
    """Barras verticales agrupadas en escala 0-1."""
    ancho, alto = 760, 340
    izq, der, arriba, abajo = 48, 16, 84, 52
    base = alto - abajo
    alto_util = base - arriba
    cuerpo = [_texto(16, 28, titulo, "t1"), _texto(16, 48, subtitulo, "t2")]
    cuerpo += _leyenda(16, 70, [(f'class="{c}"', n) for c, n, _ in series])
    for i in range(6):
        v = i / 5
        y = base - v * alto_util
        cuerpo.append(
            f'<line x1="{izq}" x2="{ancho - der}" y1="{y:.1f}" y2="{y:.1f}" class="rejilla"/>'
        )
        cuerpo.append(_texto(izq - 8, y + 4, f"{v:.1f}", "t2", "end"))
    hueco_grupo = (ancho - izq - der) / len(grupos)
    grosor = min(34.0, (hueco_grupo - 24) / len(series) - 2)
    for g, nombre in enumerate(grupos):
        total = len(series) * grosor + (len(series) - 1) * 2
        x0 = izq + g * hueco_grupo + (hueco_grupo - total) / 2
        for s, (clase, _, valores) in enumerate(series):
            v = valores[g]
            x = x0 + s * (grosor + 2)
            cuerpo.append(_barra_v(x, base, grosor, v * alto_util, clase))
            cuerpo.append(
                _texto(x + grosor / 2, base - v * alto_util - 5, formato.format(v), "val", "middle")
            )
        cuerpo.append(
            _texto(izq + g * hueco_grupo + hueco_grupo / 2, base + 20, nombre, "t2", "middle")
        )
    cuerpo.append(f'<line x1="{izq}" x2="{ancho - der}" y1="{base}" y2="{base}" class="eje"/>')
    return _svg(ancho, alto, titulo, cuerpo)


def latencias(titulo: str, subtitulo: str, filas: list[tuple[str, float, float]]) -> str:
    """Barras horizontales: medido en local frente a estimado en producción (segundos)."""
    ancho = 760
    izq, der, arriba = 210, 70, 92
    paso, grosor = 56, 18
    alto = arriba + paso * len(filas) + 4
    maximo = max(max(loc, prod) for _, loc, prod in filas)
    escala = (ancho - izq - der) / maximo
    cuerpo = [_texto(16, 28, titulo, "t1"), _texto(16, 48, subtitulo, "t2")]
    cuerpo += _leyenda(
        16,
        74,
        [
            ('class="s1"', "Medido en local (PC, CPU)"),
            ('fill="url(#estimado)" class="borde"', "Estimado en producción (GPU)"),
        ],
    )
    for i, (nombre, local, prod) in enumerate(filas):
        y = arriba + i * paso
        cuerpo.append(_texto(izq - 12, y + grosor + 2, nombre, "t2", "end"))
        cuerpo.append(_barra_h(izq, y, local * escala, grosor, 'class="s1"'))
        cuerpo.append(_texto(izq + local * escala + 6, y + 13, f"{local:g} s", "val"))
        y2 = y + grosor + 2
        cuerpo.append(
            _barra_h(izq, y2, max(prod * escala, 2), grosor, 'fill="url(#estimado)" class="borde"')
        )
        cuerpo.append(_texto(izq + max(prod * escala, 2) + 6, y2 + 13, f"~{prod:g} s", "val"))
    fondo = arriba + paso * len(filas) - 10
    cuerpo.append(f'<line x1="{izq}" x2="{izq}" y1="{arriba - 6}" y2="{fondo}" class="eje"/>')
    return _svg(ancho, alto, titulo, cuerpo)


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    graficos = {
        "recuperacion.svg": barras_agrupadas(
            "Recuperación: ¿llega el apartado correcto al contexto?",
            "33 preguntas con respuesta · el reranker es el único que mejora a la vectorial",
            ["Vectorial", "Full-text", "Híbrida (RRF)", "Híbrida + reranker"],
            [
                ("s1", "hit@1 (primer resultado)", [0.73, 0.18, 0.45, 0.67]),
                ("s2", "hit@5 (entre los 5 primeros)", [0.85, 0.48, 0.85, 0.88]),
                ("s3", "MRR@10", [0.80, 0.29, 0.61, 0.76]),
            ],
        ),
        "rag-vs-agente.svg": barras_agrupadas(
            "RAG simple frente a agente",
            "40 preguntas · la v2 solo reescribe cuando el RAG se abstiene",
            ["Corrección (juez)", "Cita correcta", "Responde cuando hay respuesta"],
            [
                ("s1", "RAG simple", [0.79, 0.70, 27 / 33]),
                ("s2", "Agente v1: reescribe siempre", [0.76, 0.70, 30 / 33]),
                ("s3", "Agente v2: reescribe si se abstiene", [0.82, 0.73, 29 / 33]),
            ],
        ),
        "latencia.svg": latencias(
            "Latencia por pregunta (p50)",
            "Local: qwen2.5:7b en Ollama. Producción: estimación para un 7B en GPU, no medida.",
            [
                ("RAG simple", 13.8, 1.5),
                ("Agente v2 sin reintento", 15.5, 1.5),
                ("Agente v2 con reintento", 28.6, 3.5),
                ("Reranker (30 pares)", 29, 0.2),
            ],
        ),
    }
    for nombre, svg in graficos.items():
        (SALIDA / nombre).write_text(svg, encoding="utf-8")
        print(f"escrito {SALIDA / nombre}")


if __name__ == "__main__":
    main()
