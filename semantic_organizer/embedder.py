"""
embedder.py
-----------
Converts extracted document text into dense semantic vectors
using Sentence-Transformers (all-MiniLM-L6-v2).

Model choice (per design doc):
  all-MiniLM-L6-v2 — good balance of speed and quality,
  optimized for English, runs fully offline after first download.
"""

import numpy as np
from pathlib import Path

# Model name as specified in the design doc
MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level model cache — load once, reuse across calls
_model = None


def get_model():
    """Load the embedding model (lazy, cached after first load)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[embedder] Loading model '{MODEL_NAME}'...")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[embedder] Model loaded.")
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of strings into a 2D numpy array of shape (N, embedding_dim).

    Args:
        texts:      List of document text strings.
        batch_size: Number of texts to encode at once (tune for memory/speed).

    Returns:
        np.ndarray of shape (N, 384) — 384 is the dimension for all-MiniLM-L6-v2.
    """
    model = get_model()
    print(f"[embedder] Embedding {len(texts)} document(s) in batches of {batch_size}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Normalize for cosine similarity
    )
    print(f"[embedder] Done. Embedding shape: {embeddings.shape}")
    return embeddings


def embed_documents(documents: list[dict], batch_size: int = 32) -> tuple[list[dict], np.ndarray]:
    """
    Embed a list of document dicts produced by extractor.extract_folder().

    Args:
        documents:  List of {"path": Path, "text": str} dicts.
        batch_size: Passed to embed_texts.

    Returns:
        (documents, embeddings)
          - documents:  Same list (unmodified, for index alignment)
          - embeddings: np.ndarray of shape (N, 384)
    """
    if not documents:
        raise ValueError("[embedder] No documents to embed.")

    texts = [doc["text"] for doc in documents]
    embeddings = embed_texts(texts, batch_size=batch_size)

    return documents, embeddings
