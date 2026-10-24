"""Streamlit UI. Run from the project root:  streamlit run frontend/app.py"""
import os
from urllib.parse import quote

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Document Intelligence Platform", page_icon="📄", layout="wide")


def api(method, path, **kwargs):
    """Call the backend. Returns (json, None) on success or (None, error_message)."""
    timeout = kwargs.pop("timeout", 600)
    try:
        response = requests.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)
    except requests.ConnectionError:
        st.error(
            f"Cannot reach the backend at {BACKEND_URL}. Start it with:\n\n"
            "`uvicorn backend.main:app --port 8000`"
        )
        st.stop()
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, str(detail)
    return response.json(), None


def render_sources(sources):
    if not sources:
        return
    st.markdown("**Sources**")
    for s in sources:
        label = f"[{s['id']}] {s['source']} — page {s['page']} (similarity {s['score']:.2f})"
        with st.expander(label):
            st.write(s["text"])


st.title("📄 Document Intelligence Platform")

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

docs, error = api("GET", "/documents")
if error:
    st.error(error)
    st.stop()
names = [d["filename"] for d in docs]

# ---------------- Sidebar: upload + document list ----------------
with st.sidebar:
    st.header("1. Add a PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded and st.button("Process & index", type="primary"):
        with st.spinner("Extracting text, chunking, embedding, storing…"):
            result, error = api(
                "POST",
                "/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
            )
        if error:
            st.error(error)
        else:
            st.session_state["flash"] = (
                f"Indexed {result['filename']}: {result['pages']} pages, "
                f"{result['chunks']} chunks."
            )
            st.rerun()

    st.header("Indexed documents")
    if not docs:
        st.caption("Nothing indexed yet.")
    for d in docs:
        left, right = st.columns([4, 1])
        left.markdown(f"**{d['filename']}**  \n{d['chunks']} chunks")
        if right.button("🗑", key=f"delete-{d['filename']}", help="Delete this document"):
            _, error = api("DELETE", f"/documents/{quote(d['filename'])}")
            if error:
                st.error(error)
            else:
                st.rerun()

# ---------------- Main area ----------------
if not docs:
    st.info("Upload a PDF in the sidebar and click **Process & index** to get started.")
    st.stop()

ask_tab, insights_tab, search_tab = st.tabs(["💬 Ask", "🧠 Insights", "🔎 Raw search"])

with ask_tab:
    scope = st.selectbox("Search in", ["All documents"] + names, key="ask_scope")
    question = st.text_input("Your question", placeholder="What are the termination clauses?")
    top_k = st.slider("Chunks to retrieve", 1, 10, 5, key="ask_k")
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving and generating (the first answer can take a minute)…"):
            data, error = api(
                "POST",
                "/ask",
                json={
                    "question": question,
                    "top_k": top_k,
                    "source": None if scope == "All documents" else scope,
                },
            )
        if error:
            st.error(error)
        else:
            st.subheader("Answer")
            st.write(data["answer"])
            render_sources(data["sources"])

with insights_tab:
    doc = st.selectbox("Document", names, key="insight_doc")
    buttons = {
        "summary": "📝 Summarize",
        "risks": "⚠️ Risks",
        "key_dates": "📅 Key dates",
        "obligations": "📌 Obligations",
    }
    columns = st.columns(len(buttons))
    chosen = None
    for column, (kind, label) in zip(columns, buttons.items()):
        if column.button(label, use_container_width=True, key=f"insight-{kind}"):
            chosen = kind
    if chosen:
        with st.spinner(f"Generating {buttons[chosen]}…"):
            data, error = api("POST", "/insights", json={"kind": chosen, "source": doc})
        if error:
            st.error(error)
        else:
            st.subheader(buttons[chosen])
            st.write(data["answer"])
            render_sources(data["sources"])

with search_tab:
    st.caption("Retrieval only, no LLM. Use this to judge whether the right chunks are found.")
    scope = st.selectbox("Search in", ["All documents"] + names, key="search_scope")
    query = st.text_input("Search query", key="search_query")
    k = st.slider("Results", 1, 20, 5, key="search_k")
    if st.button("Search") and query.strip():
        params = {"q": query, "top_k": k}
        if scope != "All documents":
            params["source"] = scope
        data, error = api("GET", "/search", params=params)
        if error:
            st.error(error)
        else:
            for hit in data["results"]:
                label = f"{hit['source']} — page {hit['page']} — similarity {hit['score']:.2f}"
                with st.expander(label, expanded=True):
                    st.write(hit["text"])
