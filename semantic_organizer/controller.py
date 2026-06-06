"""
controller.py
-------------
Executes file organization based on labeled cluster results.

Two operation modes (per design doc):
  - move: relocate original files into labeled subfolders (default)
  - copy: duplicate files into a new output folder, originals untouched

Writes a manifest.json after every commit so operations can be undone.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path


MANIFEST_FILE = "manifest.json"


def _safe_dest(dest_path: Path) -> Path:
    """
    If dest_path already exists, append a counter suffix to avoid overwriting.
    e.g. report.txt → report_1.txt → report_2.txt
    """
    if not dest_path.exists():
        return dest_path
    stem = dest_path.stem
    suffix = dest_path.suffix
    parent = dest_path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def commit(
    clusters: list[dict],
    output_dir: Path,
    mode: str = "move",
    store_dir: Path = None,
) -> Path:
    """
    Execute the file organization based on labeled cluster results.

    Args:
        clusters:   Labeled cluster list from labeler.label_clusters().
                    Each cluster: {"cluster_id": int, "label": str, "documents": [...]}
        output_dir: Root directory where labeled subfolders will be created.
                    In move mode this is the same as the source folder (files moved in place).
                    In copy mode this is a separate destination folder.
        mode:       "move" — move original files (destructive, but undoable via manifest).
                    "copy" — copy files, originals untouched.
        store_dir:  If provided, manifest.json is written here.
                    Defaults to output_dir / ".semantic_store".

    Returns:
        Path to the manifest file written.
    """
    mode = mode.lower()
    if mode not in ("move", "copy"):
        raise ValueError(f"[controller] mode must be 'move' or 'copy', got '{mode}'.")

    output_dir = Path(output_dir)
    store_dir = Path(store_dir) if store_dir else output_dir / ".semantic_store"
    store_dir.mkdir(parents=True, exist_ok=True)

    operations = []  # track every file operation for the manifest

    print(f"[controller] Mode: {mode.upper()}")
    print(f"[controller] Output: {output_dir}")
    print(f"[controller] Organizing {sum(len(c['documents']) for c in clusters)} files into {len(clusters)} folders...\n")

    for cluster in clusters:
        label = cluster["label"] or f"cluster_{cluster['cluster_id']}"
        folder = output_dir / label
        folder.mkdir(parents=True, exist_ok=True)

        for doc in cluster["documents"]:
            src = Path(doc["path"])
            if not src.exists():
                print(f"[controller] WARNING: source not found, skipping — {src}")
                continue

            dest = _safe_dest(folder / src.name)

            if mode == "move":
                shutil.move(str(src), str(dest))
            else:
                shutil.copy2(str(src), str(dest))

            operations.append({
                "filename": src.name,
                "source": str(src),
                "destination": str(dest),
                "mode": mode,
                "cluster_label": label,
            })
            print(f"[controller] {mode.upper()}  {src.name} → {label}/")

    # Write manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "output_dir": str(output_dir),
        "total_files": len(operations),
        "operations": operations,
    }
    manifest_path = store_dir / MANIFEST_FILE
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n[controller] Done. Manifest written → {manifest_path}")
    return manifest_path


def undo(store_dir: Path) -> None:
    """
    Reverse the last commit operation using the manifest.

    For 'move' operations: moves files back to their original locations.
    For 'copy' operations: deletes the copied files.

    Args:
        store_dir: Directory containing manifest.json (same as used in commit).
    """
    store_dir = Path(store_dir)
    manifest_path = store_dir / MANIFEST_FILE

    if not manifest_path.exists():
        raise FileNotFoundError(f"[controller] No manifest found at {manifest_path}. Nothing to undo.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mode = manifest["mode"]
    operations = manifest["operations"]

    print(f"[controller] Undoing {len(operations)} file operation(s) (original mode: {mode.upper()})...")

    for op in reversed(operations):  # reverse order for safety
        src = Path(op["source"])
        dest = Path(op["destination"])

        if mode == "move":
            if dest.exists():
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest), str(src))
                print(f"[controller] RESTORED  {op['filename']} → {src.parent}/")
            else:
                print(f"[controller] WARNING: destination not found, skipping — {dest}")

        elif mode == "copy":
            if dest.exists():
                dest.unlink()
                print(f"[controller] DELETED   {op['filename']} (copy removed)")
            else:
                print(f"[controller] WARNING: copy not found, skipping — {dest}")

    # Remove manifest after successful undo
    manifest_path.unlink()
    print(f"\n[controller] Undo complete. Manifest cleared.")
