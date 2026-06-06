"""
semantic_organizer
------------------
A privacy-first, offline Python library that automatically organizes
local documents into semantically meaningful folders using embeddings and clustering.

Basic usage:
    from semantic_organizer import organize
    organize("/path/to/my/documents")

With options:
    organize(
        folder="/path/to/my/documents",
        mode="copy",              # "move" (default) or "copy"
        store_dir=".my_store",    # where to persist embeddings
        force=False,              # re-embed even if store exists
        dry_run=True,             # preview clusters without touching files
    )

Undo last operation:
    from semantic_organizer import undo
    undo(store_dir="/path/to/my/documents/.semantic_store")
"""

from pathlib import Path

from .extractor import extract_folder
from .embedder import embed_documents
from .store import save, load, store_exists
from .clusterer import auto_cluster, build_clusters
from .labeler import label_clusters
from .controller import commit, undo


__version__ = "0.1.0"
__all__ = ["organize", "undo"]

DEFAULT_STORE = ".semantic_store"


def organize(
    folder: str | Path,
    mode: str = "move",
    store_dir: str | Path = None,
    force: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """
    Automatically organize documents in a folder into semantic clusters.

    Runs the full pipeline:
        Extract → Embed → Store → Cluster → Label → (Commit)

    Args:
        folder:    Path to the folder containing documents to organize.
                   Supports .txt, .pdf, and .docx files.

        mode:      How to organize files:
                   "move" — move original files into labeled subfolders (default).
                   "copy" — copy files into labeled subfolders, originals untouched.

        store_dir: Where to persist embeddings between runs.
                   Defaults to <folder>/.semantic_store.
                   Reused on subsequent runs so documents aren't re-embedded.
                   Pass force=True to rebuild the store from scratch.

        force:     If True, re-embed all documents even if a store already exists.
                   Useful when you've added or removed files from the folder.

        dry_run:   If True, run the full pipeline and print the proposed structure
                   but do NOT move or copy any files. Useful for previewing results.

    Returns:
        List of cluster dicts representing the proposed/applied structure:
        [
            {
                "cluster_id": 0,
                "label": "machine_learning_neural_network",
                "documents": [
                    {"filename": "paper.pdf", "path": "/abs/path/paper.pdf"},
                    ...
                ]
            },
            ...
        ]

    Raises:
        ValueError: If folder doesn't exist or mode is invalid.
        FileNotFoundError: If no supported documents are found.

    Examples:
        # Organize in place (move files)
        from semantic_organizer import organize
        organize("/path/to/documents")

        # Safe preview first
        clusters = organize("/path/to/documents", dry_run=True)

        # Copy mode — originals untouched
        organize("/path/to/documents", mode="copy")

        # Force re-embed after adding new files
        organize("/path/to/documents", force=True)
    """
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"[semantic_organizer] '{folder}' is not a valid directory.")

    mode = mode.lower()
    if mode not in ("move", "copy"):
        raise ValueError(f"[semantic_organizer] mode must be 'move' or 'copy', got '{mode}'.")

    store_path = Path(store_dir) if store_dir else folder / DEFAULT_STORE

    print("=" * 60)
    print("  semantic-organizer")
    print(f"  Folder : {folder}")
    print(f"  Mode   : {mode.upper()}{' (DRY RUN)' if dry_run else ''}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Stage 1 — Extract & Embed
    # ------------------------------------------------------------------ #
    if store_exists(store_path) and not force:
        print(f"\n[1/4] Existing store found — loading embeddings...")
        metadata, embeddings = load(store_path)
        print("[1/4] Re-extracting text for label generation...")
        documents = extract_folder(folder)
    else:
        print(f"\n[1/4] Extracting text from documents...")
        documents = extract_folder(folder)

        if not documents:
            raise FileNotFoundError(
                f"[semantic_organizer] No supported documents found in '{folder}'. "
                "Supported formats: .txt, .pdf, .docx"
            )

        print(f"\n[2/4] Generating embeddings...")
        documents, embeddings = embed_documents(documents)

        print(f"\n[3/4] Saving embeddings to store...")
        save(store_path, documents, embeddings)
        metadata, embeddings = load(store_path)

    # ------------------------------------------------------------------ #
    # Stage 2 — Cluster & Label
    # ------------------------------------------------------------------ #
    print(f"\n[3/4] Clustering {len(metadata)} documents...")
    labels, k, score, _ = auto_cluster(embeddings)
    clusters = build_clusters(metadata, labels, k)

    print(f"\n[4/4] Generating folder labels...")
    full_texts = {str(doc["path"].resolve()): doc["text"] for doc in documents}
    clusters = label_clusters(clusters, full_texts)

    # Deduplicate labels — if two clusters get the same label, append 2, 3 etc.
    seen: dict[str, int] = {}
    for cluster in clusters:
        label = cluster["label"]
        if label in seen:
            seen[label] += 1
            cluster["label"] = f"{label} {seen[label]}"
        else:
            seen[label] = 1

    # ------------------------------------------------------------------ #
    # Preview
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  Proposed Structure")
    print("=" * 60)
    for cluster in clusters:
        print(f"\n  📁 {cluster['label']}/")
        for doc in cluster["documents"]:
            print(f"      {doc['filename']}")

    # ------------------------------------------------------------------ #
    # Stage 3 — Commit (skipped in dry_run mode)
    # ------------------------------------------------------------------ #
    if dry_run:
        print("\n  [DRY RUN] No files were moved or copied.")
    else:
        print(f"\n  Committing ({mode.upper()})...")
        commit(clusters, output_dir=folder, mode=mode, store_dir=store_path)

    print("\n" + "=" * 60)
    print(f"  Done. {len(metadata)} documents → {k} clusters.")
    print("=" * 60)

    return clusters