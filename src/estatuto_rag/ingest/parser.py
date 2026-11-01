"""Parser del XML de la API de legislación consolidada del BOE."""

import re
import unicodedata
import xml.etree.ElementTree as ET

from estatuto_rag.ingest.models import Articulo

BLOQUES_EXCLUIDOS = frozenset({"preambulo", "aunico", "ddunica", "firma"})
CLASES_TEXTO = frozenset({"parrafo", "parrafo_2"})

CABECERA = re.compile(
    r"^(?P<tipo>Artículo|Disposición adicional|Disposición transitoria|Disposición final)"
    r" (?P<numero>[^.]+)\.\s*(?P<titulo>.*?)\.?$"
)
PREFIJOS = {
    "Disposición adicional": "da",
    "Disposición transitoria": "dt",
    "Disposición final": "df",
}

UNIDADES = {
    "primera": 1,
    "segunda": 2,
    "tercera": 3,
    "cuarta": 4,
    "quinta": 5,
    "sexta": 6,
    "septima": 7,
    "octava": 8,
    "novena": 9,
}
ESPECIALES = {"unica": 1, "decima": 10, "undecima": 11, "duodecima": 12, "vigesima": 20}


def _sin_tildes(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


def ordinal_a_numero(ordinal: str) -> int:
    """'vigesimosegunda' -> 22, 'décima' -> 10, 'primera' -> 1."""
    palabra = _sin_tildes(ordinal.lower())
    if palabra in UNIDADES:
        return UNIDADES[palabra]
    if palabra in ESPECIALES:
        return ESPECIALES[palabra]
    for prefijo, decena in (("decim", 10), ("vigesim", 20)):
        if palabra.startswith(prefijo):
            # "decimotercera" -> "otercera" -> "tercera"; "decimoctava" -> "octava"
            resto = palabra.removeprefix(prefijo)
            if resto not in UNIDADES:
                resto = resto.removeprefix("o")
            return decena + UNIDADES[resto]
    raise ValueError(f"Ordinal desconocido: {ordinal!r}")


def parsear_cabecera(cabecera: str) -> tuple[str, str]:
    """'Artículo 20 bis. Derechos…' -> ('20bis', 'Derechos…')."""
    m = CABECERA.match(cabecera)
    if m is None:
        raise ValueError(f"Cabecera no reconocida: {cabecera!r}")
    tipo, numero, titulo = m.group("tipo"), m.group("numero"), m.group("titulo")
    if tipo == "Artículo":
        return numero.replace(" ", ""), titulo
    return f"{PREFIJOS[tipo]}{ordinal_a_numero(numero)}", titulo


def _texto(p: ET.Element) -> str:
    """Texto de un <p>, incluido el de sus etiquetas hijas, con los espacios normalizados."""
    return " ".join("".join(p.itertext()).split())


def _nivel(texto: str) -> str:
    """'TÍTULO I' -> 'Título I'."""
    palabra, resto = texto.split(maxsplit=1)
    return f"{palabra.capitalize()} {resto}"


def parsear_boe(xml: str) -> list[Articulo]:
    """Convierte la respuesta XML del BOE en la lista de preceptos del Estatuto, en orden."""
    raiz = ET.fromstring(xml)
    titulo_ley: str | None = None
    capitulo: str | None = None
    seccion: str | None = None
    articulos: list[Articulo] = []

    for bloque in raiz.iter("bloque"):
        if bloque.get("id") in BLOQUES_EXCLUIDOS:
            continue
        version = bloque.findall("version")[-1]
        parrafos = version.findall("p")

        if bloque.get("tipo") == "encabezado":
            for p in parrafos:
                clase = p.get("class")
                if clase == "titulo_num":
                    titulo_ley, capitulo, seccion = _nivel(_texto(p)), None, None
                elif clase == "capitulo_num":
                    capitulo, seccion = _nivel(_texto(p)), None
                elif clase == "seccion":
                    seccion = _texto(p)
            continue

        if bloque.get("tipo") != "precepto":
            continue

        cabecera = next(_texto(p) for p in parrafos if p.get("class") == "articulo")
        numero, titulo = parsear_cabecera(cabecera)
        if not numero[0].isdigit():
            titulo_ley = capitulo = seccion = None
        texto = "\n".join(_texto(p) for p in parrafos if p.get("class") in CLASES_TEXTO)
        articulos.append(
            Articulo(
                numero=numero,
                titulo=titulo,
                texto=texto,
                titulo_ley=titulo_ley,
                capitulo=capitulo,
                seccion=seccion,
            )
        )
    return articulos
