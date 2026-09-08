"""Página del agente conversacional (Streamlit la detecta automáticamente)."""
import os

import streamlit as st

st.set_page_config(page_title="Agente — Electoral MX", page_icon="🤖", layout="wide")
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

from agente_electoral_mx import ejecutar_agente  # noqa: E402

st.title("🤖 Agente electoral")
st.caption("Dataset local primero; web (fuentes INE) después. Requiere ANTHROPIC_API_KEY "
           "en Secrets (⋮ → Settings → Secrets). TAVILY_API_KEY opcional para búsqueda web.")
if not os.environ.get("ANTHROPIC_API_KEY"):
    st.warning("Falta ANTHROPIC_API_KEY: añádela en ⋮ → Settings → Secrets de la app.")
if "msgs" not in st.session_state:
    st.session_state.msgs = []
for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
if prompt := st.chat_input("Pregunta sobre resultados distritales…"):
    st.session_state.msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            respuesta = ejecutar_agente(prompt)
        except Exception as e:
            respuesta = f"⚠️ Error: {e}"
        st.markdown(respuesta)
    st.session_state.msgs.append({"role": "assistant", "content": respuesta})
