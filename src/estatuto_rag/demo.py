"""Demo web: `uv run streamlit run src/estatuto_rag/demo.py`.

Necesita lo mismo que la API: la BD indexada (`python -m estatuto_rag.store`) y Ollama con
bge-m3 y qwen2.5:7b. Instala Streamlit con `uv sync --extra demo`.
"""

from dataclasses import dataclass

import streamlit as st

from estatuto_rag.agente.grafo import Agente
from estatuto_rag.config import cargar_config
from estatuto_rag.embeddings import OllamaEmbedder
from estatuto_rag.generation.citas import CITA, id_cita
from estatuto_rag.generation.llm import OllamaLLM
from estatuto_rag.generation.responder import Respuesta, responder
from estatuto_rag.retrieval.hibrido import RecuperadorHibrido
from estatuto_rag.retrieval.reescritura import Reescritor
from estatuto_rag.retrieval.textual import RecuperadorTextual
from estatuto_rag.retrieval.vectorial import RecuperadorVectorial
from estatuto_rag.store.db import conectar

EJEMPLOS = [
    "Me han echado por llegar tarde muchas veces, ¿pueden hacerlo?",
    "¿Cuántos días de vacaciones me corresponden?",
    "Se ha muerto un familiar mío, ¿me dan días libres?",
    "¿Cuánto cobra un camarero según el convenio de hostelería de Madrid?",
]
MODOS = {"Agente (reintenta si no lo encuentra)": "agente", "RAG simple": "rag"}


@dataclass
class Sistema:
    hibrido: RecuperadorHibrido
    llm: OllamaLLM
    agente: Agente


@st.cache_resource  # type: ignore[untyped-decorator,unused-ignore]  # sin el extra demo, st es Any
def sistema() -> Sistema:
    """Se crea una vez por proceso: conexión, modelos y grafo se reutilizan entre preguntas."""
    config = cargar_config()
    conn = conectar(config.database_url)
    embedder = OllamaEmbedder(config.ollama_url, config.embed_model)
    hibrido = RecuperadorHibrido([RecuperadorVectorial(conn, embedder), RecuperadorTextual(conn)])
    llm = OllamaLLM(config.ollama_url, config.llm_model)
    return Sistema(hibrido, llm, Agente(hibrido, llm, Reescritor(llm)))


def resaltar_citas(texto: str) -> str:
    return CITA.sub(lambda m: f"**:blue[[art. {m.group(1)}]]**", texto)


def mostrar(r: Respuesta) -> None:
    if r.abstencion:
        st.warning(r.respuesta, icon="🔎")
    else:
        st.markdown(resaltar_citas(r.respuesta))
    if r.citas_invalidas:
        st.error("Citas que no estaban en el contexto: " + ", ".join(r.citas_invalidas))

    columnas = st.columns(3)
    columnas[0].metric("Tiempo", f"{r.latencia_ms / 1000:.1f} s")
    columnas[1].metric("Búsquedas", 2 if len(r.consultas) > 1 else 1)
    columnas[2].metric("Artículos citados", len(r.citas))

    if len(r.consultas) > 1:
        st.markdown("**El agente no lo encontró a la primera y reescribió la pregunta:**")
        for consulta in r.consultas[1:]:
            st.markdown(f"- _{consulta}_")

    st.markdown("**Artículos que ha leído el modelo**")
    citados = {c.split(".")[0] for c in r.citas} | set(r.citas)
    for fuente in r.fuentes:
        marca = "✅ " if id_cita(fuente) in citados or fuente.articulo in citados else ""
        ruta = fuente.encabezado.split(" · ")  # "Estatuto · Título I · Capítulo I · Art. 1. …"
        with st.expander(f"{marca}{ruta[-1]} ({fuente.id})"):
            st.caption(" · ".join(ruta[1:-1]))
            st.markdown(fuente.texto.replace("\n", "  \n"))  # respeta los saltos de línea


def main() -> None:
    st.set_page_config(page_title="Estatuto de los Trabajadores", page_icon="⚖️")
    st.title("Pregunta al Estatuto de los Trabajadores")
    st.caption(
        "Responde citando el artículo exacto y dice cuándo la ley no lo responde. "
        "No es asesoramiento jurídico: los convenios colectivos pueden mejorar el Estatuto."
    )

    with st.sidebar:
        modo = MODOS[st.radio("Modo", list(MODOS))]
        st.markdown("**Ejemplos**")
        for ejemplo in EJEMPLOS:
            if st.button(ejemplo, use_container_width=True):
                st.session_state["pregunta"] = ejemplo

    pregunta = st.text_input("Tu pregunta", key="pregunta", max_chars=500)
    if not pregunta or len(pregunta.strip()) < 3:
        return

    s = sistema()
    with st.spinner("Buscando en el Estatuto…"):
        r = (
            s.agente.responder(pregunta)
            if modo == "agente"
            else responder(pregunta, s.hibrido, s.llm)
        )
    mostrar(r)


if __name__ == "__main__":  # streamlit run ejecuta el script como __main__
    main()
