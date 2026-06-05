"""
chunking.py — Document splitting strategies for the RAG pipeline.

Two strategies are combined here:
  1. Semantic Chunking   — splits a document at natural topic boundaries
                           by comparing sentence embeddings. When two
                           consecutive sentences become dissimilar enough,
                           a split is inserted.  These large, topic-coherent
                           pieces become the PARENT chunks.

  2. Hierarchical Chunking — each parent is further sliced into smaller
                             CHILD chunks (800 chars).  Children are what
                             gets stored in the vector database and searched.
                             When a child is retrieved, we can look up its
                             parent to get the surrounding context.

Why two levels?
  - Small children → precise embedding match (exact sentence found fast)
  - Large parents  → more context sent to the LLM (better answer quality)
"""

import uuid  # used to create a unique ID for each parent chunk

from langchain_experimental.text_splitter import SemanticChunker  # embedding-based splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter  # fixed-size splitter


def build_chunks(docs, embeddings):
    """
    Convert a list of LangChain Documents into a two-level hierarchy.

    Args:
        docs       : list of Document objects (loaded from data/ folder)
        embeddings : an embeddings model instance (used by SemanticChunker
                     to measure similarity between sentences)

    Returns:
        child_chunks  : flat list of small Document objects → go into FAISS
        parent_store  : dict  { parent_id (str) → parent Document }
    """

    # ── Level 1: Semantic splitting ──────────────────────────────────────────
    # SemanticChunker embeds every sentence, then measures cosine similarity
    # between adjacent sentences.  When similarity drops below the 85th
    # percentile threshold, it inserts a chunk boundary.
    # Result: large, topic-coherent "parent" chunks.
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",   # compare against the distribution of all similarities
        breakpoint_threshold_amount=85,           # split when similarity is in the bottom 15%
    )

    # ── Level 2: Fixed-size splitting ────────────────────────────────────────
    # Each parent is sliced into smaller children for precise retrieval.
    # chunk_size=800  → each child is at most 800 characters
    # chunk_overlap=100 → 100 characters are shared between adjacent children
    #                     so sentences don't get cut off at boundaries
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    parent_store = {}   # maps parent_id → parent Document
    child_chunks = []   # all children from all documents, stored flat

    # Split every loaded document into semantic parents first
    parents = semantic_splitter.split_documents(docs)

    for parent in parents:
        # Give each parent a unique ID so children can reference it later
        pid = str(uuid.uuid4())
        parent.metadata["doc_id"] = pid          # tag the parent itself
        parent_store[pid] = parent               # save it in the lookup dict

        # Now split this parent into children and tag each one
        for child in child_splitter.split_documents([parent]):
            child.metadata["parent_id"] = pid    # link child → parent
            child_chunks.append(child)

    return child_chunks, parent_store


def classify_query(question: str) -> str:
    """
    Decide whether a question needs a broad answer (more context)
    or a precise answer (targeted snippet).

    Returns:
        "broad"   → the question is open-ended, explanatory, or long.
                    The retriever will return the full PARENT chunk.
        "precise" → the question is short and factual.
                    The retriever will return the small CHILD chunk.

    Logic:
        - If the question starts with or contains words like "how", "why",
          "explain", "list" etc. → it needs more context → broad.
        - If the question is longer than 8 words → likely complex → broad.
        - Otherwise → precise.
    """
    q = question.lower().strip()

    # Keywords that signal the user wants an explanation, not a quick fact
    broad_signals = [
        "how", "why", "explain", "describe", "what are", "list",
        "tell me about", "summarize", "overview", "difference", "compare",
    ]

    is_broad = any(q.startswith(s) or s in q for s in broad_signals)
    is_long  = len(q.split()) > 8   # long questions tend to need more context

    return "broad" if (is_broad or is_long) else "precise"
