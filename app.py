from __future__ import annotations

import streamlit as st

from src import db, llm
from src.config import settings

st.set_page_config(page_title="DocsSearch", page_icon="💪🤖", layout="wide")
st.title("DocsSearch - Hehe Environment Checkkk")
st.caption("Hehe")

client = llm.get_client()

with st.sidebar:
    st.subheader("Config")
    st.code(
        f"LLM     {settings.llm_model}\n"
        f"EMBED   {settings.embed_model} ({settings.embed_dim}-d)\n"
        f"OLLAMA  {settings.ollama_base_url}",
        language="text",
    )
    try:
        st.write("Installed models:")
        st.write(client.list_models())
    except Exception as exc:
        st.error(f"Cannot list models: {exc}")

try:
    with db.connect() as conn:
        documents, chunks, runs = conn.execute(
            "SELECT (SELECT count(*) FROM documents),"
            "       (SELECT count(*) FROM chunks),"
            "       (SELECT count(*) FROM runs)"
        ).fetchone()
    st.success(f"Postgres ok — documents={documents}, chunks={chunks}, runs={runs}")
except Exception as exc:
    st.error(f"Postgres not reachable: {exc}")

question = st.text_input(
    "Ask the model directly (no retrieval)", "Say hello in five words."
)
if st.button("Send") and question.strip():
    with st.spinner("Calling Ollama…"):
        try:
            result = client.generate([{"role": "user", "content": question}])
            st.write(result["text"])
            st.caption(f"{result['tokens_in']} tokens in / {result['tokens_out']} out")
        except Exception as exc:
            st.error(f"Ollama call failed: {exc}")