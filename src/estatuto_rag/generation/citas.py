"""Formato del contexto, extracción y validación de citas."""

import re

from estatuto_rag.retrieval.base import Resultado

ABSTENCION = "No encuentro la respuesta en el Estatuto de los Trabajadores."
CITA = re.compile(r"\[art\. ([^\]]+)\]")


def id_cita(resultado: Resultado) -> str:
    """Lo que se cita: '38', '38.1'… (las partes '41.4#2' se citan como '41.4')."""
    if resultado.apartado is None:
        return resultado.articulo
    return f"{resultado.articulo}.{resultado.apartado}"


def formatear_contexto(resultados: list[Resultado]) -> str:
    bloques = [
        f'<articulo id="{id_cita(r)}">\n{r.encabezado}\n{r.texto}\n</articulo>' for r in resultados
    ]
    return "\n\n".join(bloques)


def extraer_citas(texto: str) -> list[str]:
    """Citas únicas en orden de aparición. Acepta '[art. 38.1]' y '[art. 38.1, 38.2]'."""
    citas: list[str] = []
    for grupo in CITA.findall(texto):
        for cita in grupo.split(","):
            cita = cita.strip()
            if cita and cita not in citas:
                citas.append(cita)
    return citas


def es_abstencion(texto: str) -> bool:
    return ABSTENCION.lower().rstrip(".") in texto.lower()
