"""Parser del XML de la API de legislación consolidada del BOE.

Formato (ver tests/fixtures/boe_a38.xml y el spike):
- <response><data> contiene <bloque id=… tipo="encabezado"|"precepto"|…> en orden del documento.
- Cada bloque tiene una o varias <version>; la vigente es la ÚLTIMA.
- En un precepto: <p class="articulo"> es "Artículo 38. Vacaciones anuales.", los
  <p class="parrafo"> son el texto y <p class="nota_pie"> son notas de modificación (se descartan).
- Los encabezados (ti, cii, s5…) fijan Título / Capítulo / Sección de los preceptos que siguen.
- OJO: los ids de bloque NO sirven como número de artículo ("a2-2" es el art. 20 bis, "da" es la
  disposición adicional vigesimosegunda). El número sale del texto de <p class="articulo">.
- Las letras a), b)… vienen como <p class="parrafo_2"> y son texto del precepto.
- Antes de las disposiciones no hay encabezado que "cierre" el último Título: hay que resetear la
  jerarquía al llegar a la primera disposición.
"""

from estatuto_rag.ingest.models import Articulo

# Bloques que pertenecen al Real Decreto Legislativo que aprueba el Estatuto, no al Estatuto.
BLOQUES_EXCLUIDOS = frozenset({"preambulo", "aunico", "ddunica", "firma"})


def parsear_boe(xml: str) -> list[Articulo]:
    """Convierte la respuesta XML del BOE en la lista de preceptos del Estatuto, en orden."""
    raise NotImplementedError
