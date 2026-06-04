import io
import streamlit as st
from pathlib import Path
from docx import Document as DocxDocument

from rag_graph import search, build_index, init_index

DATA_DIR  = Path(__file__).parent / "data"
SUPPORTED = {".txt", ".pdf", ".docx"}
MEMORY_TURNS = 6


def _format_history(messages: list) -> str:
    recent = messages[-MEMORY_TURNS:]
    if not recent:
        return ""
    lines = ["Chat History:"]
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines) + "\n\n"


def _export_docx(messages: list) -> io.BytesIO:
    doc = DocxDocument()
    doc.add_heading("Chat Export", level=0)
    for m in messages:
        role = "You" if m["role"] == "user" else "Assistant"
        p = doc.add_paragraph()
        p.add_run(f"{role}: ").bold = True
        p.add_run(m["content"])
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbot")
st.caption("Ask questions about your documents. Powered by Groq · Llama-3.3-70b")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Documents")
    files = [f for f in DATA_DIR.rglob("*") if f.suffix.lower() in SUPPORTED]
    if files:
        st.success(f"{len(files)} document(s) indexed")
        for f in files:
            st.text(f"• {f.name}")
    else:
        st.warning("No documents found.\nAdd .txt / .pdf / .docx files to data/")

    st.divider()
    if st.button("🔄 Rebuild Index", use_container_width=True):
        with st.spinner("Rebuilding index..."):
            try:
                build_index(force=True)
                init_index(force=True)
                st.success("Index rebuilt!")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.divider()
    if st.session_state.get("messages"):
        st.download_button(
            label="📄 Export Chat (.docx)",
            data=_export_docx(st.session_state.messages),
            file_name="chat_export.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        "**Pipeline**\n"
        "- Semantic + Hierarchical chunking (800 chars)\n"
        "- all-MiniLM-L6-v2 embeddings\n"
        "- FAISS dense retrieval (k=10)\n"
        "- BM25 reranking (top 5)\n"
        "- Groq Llama-3.3-70b\n"
        "- Conversational memory\n"
        "- DOCX export"
    )

# ── Lazy-init index ───────────────────────────────────────────────────────────
if "ready" not in st.session_state:
    with st.spinner("Loading index... (first run builds embeddings, ~30 s)"):
        try:
            init_index()
            st.session_state.ready = True
        except Exception as e:
            st.error(f"Failed to load index: {e}\n\nAdd documents to data/ then click Rebuild Index.")
            st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

def _render_chunks(chunks: list):
    """Render clickable file links + exact chunk content in an expander."""
    if not chunks:
        return
    with st.expander(f"📄 Referenced chunks ({len(chunks)})"):
        for i, chunk in enumerate(chunks, 1):
            fname = chunk["filename"] or "Unknown"
            page  = chunk.get("page")
            uri   = chunk.get("file_uri", "")

            # Clickable link if we have a valid file URI
            header = f"**Chunk {i} —** "
            if uri:
                header += f"[{fname}]({uri})"
            else:
                header += f"`{fname}`"
            if page is not None:
                header += f"  •  page {int(page) + 1}"

            st.markdown(header, unsafe_allow_html=True)
            st.markdown(
                f"<div style='background:#f0f2f6;border-left:3px solid #4a90d9;"
                f"padding:8px 12px;border-radius:4px;font-size:0.85em;"
                f"white-space:pre-wrap'>{chunk['content']}</div>",
                unsafe_allow_html=True,
            )
            if i < len(chunks):
                st.divider()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chunks"):
            _render_chunks(msg["chunks"])
        if msg.get("meta"):
            st.caption(msg["meta"])

# ── Input ─────────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                history = _format_history(st.session_state.messages[:-1])
                result = search(question, chat_history=history)
                answer = result["answer"]
                chunks = result["chunks"]
                qtype  = result["query_type"]
                meta   = (
                    f"Query: **{qtype}** | "
                    f"{'parent chunks' if qtype == 'broad' else 'child chunks'} + BM25 rerank"
                )
            except Exception as e:
                answer, chunks, meta = f"Error: {e}", [], ""

        st.markdown(answer)
        _render_chunks(chunks)
        if meta:
            st.caption(meta)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": chunks,
        "meta": meta,
    })
