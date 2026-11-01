"""Parser del XML de la API de legislación consolidada del BOE.

Formato (ver tests/fixtures/boe_a38.xml y el spike):
- <response><data> contiene <bloque id=… tipo="encabezado"|"precepto"|…> en orden del documento.
- Cada bloque tiene una o varias <version>; la vigente es la ÚLTIMA.
- En un precepto: <p class="articulo"> es "Artículo 38. Vacaciones anuales.", los
  <p class="parrafo"> son el texto y <p class="nota_pie"> son notas de modificación (se descartan).
- Los encabezados (ti, cii, s5…) fijan Título / Capítulo / Sección de los preceptos que siguen.
"""

from estatuto_rag.ingest.models import Articulo

# Bloques que pertenecen al Real Decreto Legislativo que aprueba el Estatuto, no al Estatuto.
BLOQUES_EXCLUIDOS = frozenset({"preambulo", "aunico", "ddunica", "firma"})


def parsear_boe(xml: str) -> list[Articulo]:
    """Convierte la respuesta XML del BOE en la lista de preceptos del Estatuto, en orden."""
    raise NotImplementedError
