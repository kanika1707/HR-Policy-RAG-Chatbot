import pickle
from pathlib import Path
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from chunking import build_chunks, classify_query

load_dotenv()

DATA_DIR  = Path(__file__).parent / "data"
INDEX_DIR = Path(__file__).parent / "index"

LOADERS = {".txt": TextLoader, ".pdf": PyPDFLoader, ".docx": Docx2txtLoader}

PROMPT = ChatPromptTemplate.from_template(
    """You are a knowledgeable assistant. Answer the question specifically and thoroughly using only the context below.

Rules:
- Give a direct, detailed answer — do NOT just describe what the document is about
- Cover all relevant points mentioned in the context
- Use bullet points or numbered lists when listing multiple items
- If the answer is not in the context, say "I don't have that information in the provided documents."

{chat_history}
Context:
{context}

Question: {question}
Answer:"""
)


def _embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def _load_docs():
    docs = []
    for f in DATA_DIR.rglob("*"):
        cls = LOADERS.get(f.suffix.lower())
        if cls:
            try:
                docs.extend(cls(str(f)).load())
                print(f"Loaded: {f.name}")
            except Exception as e:
                print(f"Skipped {f.name}: {e}")
    return docs


def build_index(force: bool = False):
    INDEX_DIR.mkdir(exist_ok=True)
    faiss_path = INDEX_DIR / "faiss_child"
    store_path = INDEX_DIR / "parents.pkl"
    emb = _embeddings()

    if faiss_path.exists() and store_path.exists() and not force:
        print("Loading existing index...")
        vs = FAISS.load_local(str(faiss_path), emb, allow_dangerous_deserialization=True)
        with open(store_path, "rb") as fh:
            parent_store = pickle.load(fh)
        return vs, parent_store

    print("Building index...")
    docs = _load_docs()
    if not docs:
        raise ValueError(f"No supported documents found in {DATA_DIR}")

    child_chunks, parent_store = build_chunks(docs, emb)
    print(f"Parents: {len(parent_store)}  |  Children: {len(child_chunks)}")

    vs = FAISS.from_documents(child_chunks, emb)
    vs.save_local(str(faiss_path))
    with open(store_path, "wb") as fh:
        pickle.dump(parent_store, fh)

    print("Index saved.")
    return vs, parent_store


def _bm25_rerank(query: str, docs: list, top_k: int = 5) -> list:
    """Step F: re-score dense-retrieved candidates with BM25 keyword matching."""
    tokenized = [d.page_content.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def _retrieve(question: str, vs, parent_store) -> tuple:
    """
    Returns (context_string, chunks_metadata, query_type)
    Steps D -> E -> F: dense retrieval -> parent/child swap -> BM25 rerank
    """
    qtype = classify_query(question)
    candidates = vs.similarity_search(question, k=10)

    if qtype == "broad":
        seen, context_docs = set(), []
        for r in candidates:
            pid = r.metadata.get("parent_id")
            if pid and pid not in seen:
                seen.add(pid)
                context_docs.append(parent_store.get(pid, r))
    else:
        context_docs = candidates

    reranked = _bm25_rerank(question, context_docs, top_k=5)

    # Build serialisable chunk metadata for the UI
    chunks = []
    seen_content = set()
    for doc in reranked:
        content = doc.page_content.strip()
        if content in seen_content:
            continue
        seen_content.add(content)
        source_path = doc.metadata.get("source", "")
        chunks.append({
            "content":  content,
            "source":   source_path,
            "filename": Path(source_path).name if source_path else "Unknown",
            "page":     doc.metadata.get("page"),        # present for PDFs
            "file_uri": Path(source_path).as_uri() if source_path else "",
        })

    context = "\n\n---\n\n".join(c["content"] for c in chunks)
    return context, chunks, qtype


_cache: dict = {}


def init_index(force: bool = False):
    if not _cache or force:
        vs, parent_store = build_index(force=force)
        _cache["vs"] = vs
        _cache["ps"] = parent_store
        _cache["llm_chain"] = (
            PROMPT
            | ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
            | StrOutputParser()
        )


def search(question: str, chat_history: str = "") -> dict:
    """
    Main entry point.
    Returns {"answer": str, "chunks": list[dict], "query_type": str}
    Each chunk dict: {content, source, filename, page, file_uri}
    """
    init_index()
    context, chunks, qtype = _retrieve(question, _cache["vs"], _cache["ps"])
    answer = _cache["llm_chain"].invoke({
        "question":     question,
        "chat_history": chat_history,
        "context":      context,
    })
    return {"answer": answer, "chunks": chunks, "query_type": qtype}
