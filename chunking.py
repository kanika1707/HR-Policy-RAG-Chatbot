import uuid
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_chunks(docs, embeddings):
    """
    Two-level hierarchical chunking:
      Level 1 — parents: SemanticChunker splits at natural topic boundaries
                          (uses embedding similarity to find breakpoints)
      Level 2 — children: small fixed chunks (350 chars) for precise retrieval

    Each child carries a parent_id so we can fetch the full parent for context.
    Returns: (child_chunks, parent_store dict)
    """
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    parent_store = {}
    child_chunks = []

    parents = semantic_splitter.split_documents(docs)
    for parent in parents:
        pid = str(uuid.uuid4())
        parent.metadata["doc_id"] = pid
        parent_store[pid] = parent

        for child in child_splitter.split_documents([parent]):
            child.metadata["parent_id"] = pid
            child_chunks.append(child)

    return child_chunks, parent_store


def classify_query(question: str) -> str:
    """
    Detect whether the query needs broad context or a precise snippet.

    broad  → return the parent chunk (larger context window)
    precise → return the child chunk (targeted snippet)
    """
    q = question.lower().strip()
    broad_signals = [
        "how", "why", "explain", "describe", "what are", "list",
        "tell me about", "summarize", "overview", "difference", "compare",
    ]
    is_broad = any(q.startswith(s) or s in q for s in broad_signals)
    is_long = len(q.split()) > 8
    return "broad" if (is_broad or is_long) else "precise"
