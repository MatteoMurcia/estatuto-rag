"""Fidelidad numérica: ¿todos los números de la respuesta aparecen en el contexto?

En el Estatuto casi todo lo que importa es una cantidad (días, meses, horas, mensualidades),
y es justo lo que un LLM pequeño "rellena" de memoria. Esta comprobación es determinista:
no depende de un juez LLM. Limitación: detecta cifras inventadas, no cifras reales
atribuidas al supuesto equivocado.
"""

import re
import unicodedata

UNIDADES = {
    "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
    "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18,
    "diecinueve": 19, "veinte": 20, "veintiun": 21, "veintiuno": 21, "veintiuna": 21,
    "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25,
    "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
}  # fmt: skip
DECENAS = {
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
    "ochenta": 80, "noventa": 90,
}  # fmt: skip
CENTENAS = {
    "cien": 100, "ciento": 100, "doscientos": 200, "doscientas": 200, "trescientos": 300,
    "trescientas": 300, "cuatrocientos": 400, "cuatrocientas": 400, "quinientos": 500,
    "quinientas": 500, "seiscientos": 600, "seiscientas": 600, "setecientos": 700,
    "setecientas": 700, "ochocientos": 800, "ochocientas": 800, "novecientos": 900,
    "novecientas": 900,
}  # fmt: skip
PALABRAS = UNIDADES | DECENAS | CENTENAS

# Referencias a preceptos: sus números no son datos ("[art. 38.1]", "artículo 34.7", "apartado 3").
REFERENCIAS = re.compile(
    r"\[art\.[^\]]*\]|\b(?:art[ií]culos?|art\.|apartados?|disposici[oó]n\s+\w+)\s+[\w.,\s]*?\d[\w.]*",
    re.IGNORECASE,
)


def _sin_tildes(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()


def numeros(texto: str) -> set[int]:
    """Valores numéricos del texto, en cifras ('30', '2012') o en letra ('treinta y tres')."""
    texto = _sin_tildes(REFERENCIAS.sub(" ", texto))
    valores = {int(n) for n in re.findall(r"\d+", texto)}
    actual: int | None = None
    for token in re.findall(r"[a-z]+", texto):
        if token in PALABRAS:
            actual = (actual or 0) + PALABRAS[token]
        elif token == "mil":
            actual = (actual or 1) * 1000
        elif token == "y" and actual is not None:
            continue  # "treinta y tres"
        else:
            if actual is not None:
                valores.add(actual)
            actual = None
    if actual is not None:
        valores.add(actual)
    return valores


def numeros_no_respaldados(respuesta: str, contexto: str) -> set[int]:
    # "un"/"una" son artículos casi siempre ("un año"): el 1 no cuenta como dato.
    return numeros(respuesta) - numeros(contexto) - {1}
