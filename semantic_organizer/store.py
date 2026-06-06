"""
store.py
--------
Persists document embeddings and metadata to disk using:
  - NumPy (.npy)  → the embedding matrix
  - JSON          → file paths and associated metadata

Storage format (in the chosen output directory):
  embeddings.npy   — float32 array of shape (N, 384)
  metadata.json    — list of dicts, one per document, index-aligned with embeddings

This simple approach means index i in embeddings.npy always corresponds
to index i in metadata.json — no database needed.
"""

import json
import numpy as np
from pathlib import Path


EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.json"


def save(store_dir: Path, documents: list[dict], embeddings: np.ndarray) -> None:
    """
    Save embeddings and metadata to store_dir.

    Args:
        store_dir:   Directory to write files into (created if it doesn't exist).
        documents:   List of {"path": Path, "text": str} dicts from extractor.
        embeddings:  np.ndarray of shape (N, 384) from embedder.
    """
    if len(documents) != len(embeddings):
        raise ValueError(
            f"[store] Mismatch: {len(documents)} documents but {len(embeddings)} embeddings."
        )

    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    # Save embedding matrix
    emb_path = store_dir / EMBEDDINGS_FILE
    np.save(str(emb_path), embeddings.astype(np.float32))
    print(f"[store] Saved embeddings → {emb_path}")

    # Build metadata — store path as string for JSON serialisation
    metadata = [
        {
            "index": i,
            "filename": doc["path"].name,
            "path": str(doc["path"].resolve()),
            "char_count": len(doc["text"]),
        }
        for i, doc in enumerate(documents)
    ]

    meta_path = store_dir / METADATA_FILE
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[store] Saved metadata  → {meta_path}")
    print(f"[store] Store contains {len(metadata)} document(s).")


def load(store_dir: Path) -> tuple[list[dict], np.ndarray]:
    """
    Load embeddings and metadata from store_dir.

    Returns:
        (metadata, embeddings)
          - metadata:   List of dicts (index-aligned with embeddings)
          - embeddings: np.ndarray of shape (N, 384)

    Raises:
        FileNotFoundError if either file is missing.
    """
    store_dir = Path(store_dir)
    emb_path = store_dir / EMBEDDINGS_FILE
    meta_path = store_dir / METADATA_FILE

    if not emb_path.exists():
        raise FileNotFoundError(f"[store] Embeddings file not found: {emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"[store] Metadata file not found: {meta_path}")

    embeddings = np.load(str(emb_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    print(f"[store] Loaded {len(metadata)} document(s) from {store_dir}")
    print(f"[store] Embedding shape: {embeddings.shape}")

    return metadata, embeddings


def store_exists(store_dir: Path) -> bool:
    """Return True if a valid store already exists at store_dir."""
    store_dir = Path(store_dir)
    return (store_dir / EMBEDDINGS_FILE).exists() and (store_dir / METADATA_FILE).exists()
