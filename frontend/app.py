"""Streamlit UI.  Run from the project root:  streamlit run frontend/app.py

By default everything runs inside this one process (no separate backend needed).
Set BACKEND_URL=http://127.0.0.1:8000 to use the FastAPI server instead.
"""
import os
import sys
from pathlib import Path

# Make `backend` importable no matter where Streamlit was started from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

st.set_page_config(page_title="Document Intelligence Platform", page_icon="📄", layout="wide")

# Streamlit Cloud keeps settings in "secrets". Copy them into environment
# variables BEFORE the backend reads its configuration.
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, (str, int, float, bool)):
            os.environ.setdefault(_key, str(_value))
except Exception:  # no secrets file: perfectly normal when running locally
    pass

from backend.config import DEMO_MAX_QUESTIONS, DEMO_MODE, PROTECTED_DOCS  # noqa: E402
from frontend.client import HttpClient, LocalClient  # noqa: E402


@st.cache_resource(show_spinner="Loading the embedding model (first start takes a minute)…")
def get_client():
    backend_url = os.getenv("BACKEND_URL")
    return HttpClient(backend_url) if backend_url else LocalClient()


def render_sources(sources):
    if not sources:
        return
    st.markdown("**Sources**")
    for s in sources:
        label = f"[{s['id']}] {s['source']} — page {s['page']} (similarity {s['score']:.2f})"
        with st.expander(label):
            st.write(s["text"])


def can_ask():
    """Demo mode: cap how many LLM calls one visitor can make (protects the free quota)."""
    if not DEMO_MODE:
        return True
    if st.session_state.get("questions_used", 0) >= DEMO_MAX_QUESTIONS:
        st.warning(
            f"Demo limit reached ({DEMO_MAX_QUESTIONS} requests per visit). "
            "Reload the page to start again, or run the project locally."
        )
        return False
    st.session_state["questions_used"] = st.session_state.get("questions_used", 0) + 1
    return True


client = get_client()
status, status_error = client.status()
if status_error:
    st.error(status_error)
    st.stop()
MAX_MB = status.get("max_upload_mb", 50)

st.title("📄 Document Intelligence Platform")
st.caption(
    "Ask questions about your PDFs and get answers with page citations, "
    "or generate a summary, risks, key dates and obligations."
)
if DEMO_MODE:
    st.info(
        "**Public demo.** The sample contract is pre-loaded, so you can start asking right away. "
        "This deployment is shared: anything you upload can be seen by other visitors and is "
        "wiped on restart, so please don't upload confidential files."
    )

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

docs, error = client.documents()
if error:
    st.error(error)
    st.stop()
names = [d["filename"] for d in docs]

# ---------------- Sidebar: upload + document list ----------------
with st.sidebar:
    st.header("1. Add a PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    st.caption(f"Text-based PDFs only (no scans). Max {MAX_MB} MB.")
    if uploaded and st.button("Process & index", type="primary"):
        with st.spinner("Extracting text, chunking, embedding, storing…"):
            result, error = client.upload(uploaded.name, uploaded.getvalue())
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
        if d["filename"] in PROTECTED_DOCS:
            right.caption("🔒")
        elif right.button("🗑", key=f"delete-{d['filename']}", help="Delete this document"):
            _, error = client.delete(d["filename"])
            if error:
                st.error(error)
            else:
                st.rerun()

    st.divider()
    st.caption(
        f"Embeddings: {status.get('embedding_backend', '?')}  \n"
        f"LLM: {status.get('llm', '?')}  \n"
        f"Mode: {client.mode}"
    )

# ---------------- Main area ----------------
if not docs:
    st.info("Upload a PDF in the sidebar and click **Process & index** to get started.")
    st.stop()

ask_tab, insights_tab, search_tab = st.tabs(["💬 Ask", "🧠 Insights", "🔎 Raw search"])

with ask_tab:
    scope = st.selectbox("Search in", ["All documents"] + names, key="ask_scope")
    question = st.text_input("Your question", placeholder="What are the termination clauses?")
    top_k = st.slider("Chunks to retrieve", 1, 10, 5, key="ask_k")
    if st.button("Ask", type="primary") and question.strip() and can_ask():
        with st.spinner("Retrieving and generating…"):
            data, error = client.ask(
                question, top_k, None if scope == "All documents" else scope
            )
        if error:
            st.error(error)
        else:
            st.subheader("Answer")
            st.markdown(data["answer"])
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
    if chosen and can_ask():
        with st.spinner(f"Generating {buttons[chosen]}…"):
            data, error = client.insight(chosen, doc)
        if error:
            st.error(error)
        else:
            st.subheader(buttons[chosen])
            st.markdown(data["answer"])
            render_sources(data["sources"])
    st.caption("AI-generated reading aid, not legal advice. Check the cited sources.")

with search_tab:
    st.caption("Retrieval only, no LLM. Use this to judge whether the right chunks are found.")
    scope = st.selectbox("Search in", ["All documents"] + names, key="search_scope")
    query = st.text_input("Search query", key="search_query")
    k = st.slider("Results", 1, 20, 5, key="search_k")
    if st.button("Search") and query.strip():
        data, error = client.search(query, k, None if scope == "All documents" else scope)
        if error:
            st.error(error)
        else:
            for hit in data["results"]:
                label = f"{hit['source']} — page {hit['page']} — similarity {hit['score']:.2f}"
                with st.expander(label, expanded=True):
                    st.write(hit["text"])
