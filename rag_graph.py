"""
rag_graph.py — Core RAG (Retrieval-Augmented Generation) pipeline.

Full pipeline executed on every user question:

  Step A  Load documents (.txt / .pdf / .docx) from the data/ folder
  Step B  Chunk them using semantic + hierarchical splitting  (chunking.py)
  Step C  Embed each child chunk with all-MiniLM-L6-v2
  Step D  Store child embeddings in a local FAISS vector database
  Step E  At query time: dense similarity search (top-10 candidates)
  Step F  Swap children → parents for broad queries, then BM25 rerank top-5
  Step G  Send final context + question to Groq (Llama-3.3-70b) for an answer
  Step H  Conversational memory: last 6 messages are included in the prompt
"""

import pickle                         # saves/loads the parent_store dict to disk
from pathlib import Path
from dotenv import load_dotenv        # reads GROQ_API_KEY from the .env file
from rank_bm25 import BM25Okapi       # keyword-based reranker (no neural net needed)

# Document loaders — each handles a different file format
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

# FAISS — Facebook AI Similarity Search, a fast local vector database
from langchain_community.vectorstores import FAISS

# Embedding model — converts text to a numerical vector
# all-MiniLM-L6-v2 is small (~80 MB), fast, and runs entirely on CPU
from langchain_community.embeddings import HuggingFaceEmbeddings

# Groq client — sends the prompt to Llama running on Groq's fast inference servers
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate   # structures the LLM prompt
from langchain_core.output_parsers import StrOutputParser  # extracts plain text from LLM response

from chunking import build_chunks, classify_query  # our custom chunking logic

# Load GROQ_API_KEY from .env into the environment so langchain_groq can find it
load_dotenv()

# ── Directory paths ───────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).parent / "data"    # where users put their documents
INDEX_DIR = Path(__file__).parent / "index"   # where FAISS index is saved to disk

# Maps file extension → the right loader class
LOADERS = {
    ".txt":  TextLoader,
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
}

# ── LLM Prompt Template ───────────────────────────────────────────────────────
# {chat_history} → last few messages for conversational memory (Step H)
# {context}      → the retrieved document chunks (Steps E/F)
# {question}     → the user's current question
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
    """
    Load the HuggingFace sentence embedding model.
    'all-MiniLM-L6-v2' maps text → 384-dimensional vector.
    It runs locally on CPU — no API key or internet needed after first download.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def _load_docs():
    """
    Walk the data/ folder recursively and load every supported file.
    Returns a flat list of LangChain Document objects.
    Each Document has .page_content (text) and .metadata (source path, page, etc.)
    """
    docs = []
    for f in DATA_DIR.rglob("*"):                   # rglob → search subfolders too
        cls = LOADERS.get(f.suffix.lower())          # pick the right loader by extension
        if cls:
            try:
                docs.extend(cls(str(f)).load())      # load() returns a list of Documents
                print(f"Loaded: {f.name}")
            except Exception as e:
                print(f"Skipped {f.name}: {e}")      # skip corrupted / unreadable files
    return docs


def build_index(force: bool = False):
    """
    Build (or reload) the FAISS vector index and the parent_store.

    If an index already exists on disk and force=False, it is loaded from disk
    (fast, no re-embedding needed).  Pass force=True to rebuild from scratch,
    e.g. when new documents are added to data/.

    Saves two artefacts to index/:
      faiss_child/  → FAISS index of child chunk embeddings
      parents.pkl   → pickle of the {parent_id: Document} dict

    Returns:
        vs           : FAISS vectorstore object
        parent_store : dict { parent_id → parent Document }
    """
    INDEX_DIR.mkdir(exist_ok=True)
    faiss_path = INDEX_DIR / "faiss_child"
    store_path = INDEX_DIR / "parents.pkl"
    emb = _embeddings()

    # ── Load existing index from disk (fast path) ────────────────────────────
    if faiss_path.exists() and store_path.exists() and not force:
        print("Loading existing index...")
        # allow_dangerous_deserialization=True is required by FAISS when loading
        # a locally saved index (it uses pickle internally)
        vs = FAISS.load_local(str(faiss_path), emb, allow_dangerous_deserialization=True)
        with open(store_path, "rb") as fh:
            parent_store = pickle.load(fh)
        return vs, parent_store

    # ── Build fresh index (slow path — embeds every chunk) ──────────────────
    print("Building index...")
    docs = _load_docs()
    if not docs:
        raise ValueError(f"No supported documents found in {DATA_DIR}")

    # Chunk the documents using semantic + hierarchical splitting
    child_chunks, parent_store = build_chunks(docs, emb)
    print(f"Parents: {len(parent_store)}  |  Children: {len(child_chunks)}")

    # Embed all child chunks and store in FAISS
    vs = FAISS.from_documents(child_chunks, emb)

    # Persist both artefacts so we don't need to rebuild on the next run
    vs.save_local(str(faiss_path))
    with open(store_path, "wb") as fh:
        pickle.dump(parent_store, fh)

    print("Index saved.")
    return vs, parent_store


def _bm25_rerank(query: str, docs: list, top_k: int = 5) -> list:
    """
    Step F — BM25 Reranking.

    BM25 (Best Match 25) is a classic keyword-scoring algorithm.
    It scores each document by how many query words appear in it,
    weighted by term frequency and document length.

    Why rerank after FAISS?
      FAISS finds semantically similar chunks (good for paraphrases)
      but BM25 ensures the retrieved chunks also contain the exact
      keywords from the question — catching cases where semantic
      similarity alone misses the right chunk.

    Args:
        query  : the user's question
        docs   : candidate documents from FAISS (or parent swap)
        top_k  : how many to keep after reranking

    Returns:
        top_k documents sorted by BM25 score (highest first)
    """
    # Tokenise: lowercase and split on whitespace
    tokenized = [d.page_content.lower().split() for d in docs]

    # Build BM25 index over the candidate docs
    bm25 = BM25Okapi(tokenized)

    # Score all candidates against the query
    scores = bm25.get_scores(query.lower().split())

    # Sort by score descending and return top_k
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def _retrieve(question: str, vs, parent_store) -> tuple:
    """
    Steps D → E → F  —  Retrieve the most relevant document chunks.

    1. Dense retrieval (FAISS): embed the question and find the 10 most
       similar child chunks by cosine similarity.

    2. Parent/child swap (query-adaptive):
       - broad query  → look up the parent of each child, return the full
                        parent for richer context sent to the LLM.
       - precise query → keep the child chunks as-is (targeted snippets).

    3. BM25 rerank: re-score the candidates with keyword matching and
       return the top-5.

    Returns:
        context : str   — chunks joined with "---" separators, sent to LLM
        chunks  : list  — serialisable metadata dicts shown in the UI
        qtype   : str   — "broad" or "precise" (shown as a caption in the UI)
    """

    # Classify the question to decide which chunk level to return
    qtype = classify_query(question)

    # Step E: FAISS dense similarity search — fetch top-10 child chunks
    candidates = vs.similarity_search(question, k=10)

    if qtype == "broad":
        # For broad questions: swap each child for its parent chunk.
        # Use a set to deduplicate — multiple children may share the same parent.
        seen, context_docs = set(), []
        for r in candidates:
            pid = r.metadata.get("parent_id")
            if pid and pid not in seen:
                seen.add(pid)
                # Fallback to the child itself if parent_id is missing somehow
                context_docs.append(parent_store.get(pid, r))
    else:
        # For precise questions: keep the small child chunks
        context_docs = candidates

    # Step F: BM25 rerank to ensure keyword relevance, keep top-5
    reranked = _bm25_rerank(question, context_docs, top_k=5)

    # Build serialisable dicts so the UI can show file links and chunk content
    chunks = []
    seen_content = set()
    for doc in reranked:
        content = doc.page_content.strip()
        if content in seen_content:
            continue                             # skip exact duplicate content
        seen_content.add(content)

        source_path = doc.metadata.get("source", "")
        chunks.append({
            "content":  content,
            "source":   source_path,
            "filename": Path(source_path).name if source_path else "Unknown",
            "page":     doc.metadata.get("page"),          # page number (PDFs only)
            "file_uri": Path(source_path).as_uri() if source_path else "",  # clickable link
        })

    # Join chunks with a visible separator so the LLM knows where one ends
    context = "\n\n---\n\n".join(c["content"] for c in chunks)
    return context, chunks, qtype


# ── Module-level cache ────────────────────────────────────────────────────────
# Streamlit re-runs the entire script on every user interaction.
# Storing the index and chain here means they are only built once per session,
# not on every message.
_cache: dict = {}


def init_index(force: bool = False):
    """
    Load (or build) the FAISS index and construct the LLM chain.
    Results are stored in _cache so this is effectively a one-time cost.

    Pass force=True to rebuild the index from scratch (e.g. after adding
    new documents to data/).
    """
    if not _cache or force:
        vs, parent_store = build_index(force=force)
        _cache["vs"] = vs
        _cache["ps"] = parent_store

        # Build the LLM chain: PROMPT → ChatGroq → plain string output
        # temperature=0 → deterministic answers (no randomness)
        _cache["llm_chain"] = (
            PROMPT
            | ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
            | StrOutputParser()   # extracts the text content from the LLM response object
        )


def search(question: str, chat_history: str = "") -> dict:
    """
    Main public entry point — called by the Streamlit UI on every message.

    Orchestrates the full pipeline:
      retrieve relevant chunks → inject into prompt → call LLM → return result

    Args:
        question     : the user's current message
        chat_history : formatted string of recent conversation turns (Step H)

    Returns a dict with:
        "answer"     : the LLM's response (plain text)
        "chunks"     : list of chunk metadata dicts (for the UI source viewer)
        "query_type" : "broad" or "precise" (shown as a caption)
    """
    init_index()   # no-op if already initialised

    # Retrieve the best matching context and its metadata
    context, chunks, qtype = _retrieve(question, _cache["vs"], _cache["ps"])

    # Send everything to the LLM and get back a plain-text answer
    answer = _cache["llm_chain"].invoke({
        "question":     question,
        "chat_history": chat_history,  # gives the LLM memory of recent turns
        "context":      context,       # the retrieved document chunks
    })

    return {"answer": answer, "chunks": chunks, "query_type": qtype}
