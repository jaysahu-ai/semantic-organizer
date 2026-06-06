"""
examples/basic_usage.py
-----------------------
Demonstrates common usage patterns for semantic-organizer.
"""

from semantic_organizer import organize, undo


# ------------------------------------------------------------------ #
# Example 1 — Basic usage (move files in place)
# ------------------------------------------------------------------ #
organize("/path/to/my/documents")


# ------------------------------------------------------------------ #
# Example 2 — Safe preview before committing
# ------------------------------------------------------------------ #
clusters = organize("/path/to/my/documents", dry_run=True)

print("\nProposed structure:")
for cluster in clusters:
    print(f"\n📁 {cluster['label']}")
    for doc in cluster["documents"]:
        print(f"   {doc['filename']}")


# ------------------------------------------------------------------ #
# Example 3 — Copy mode (originals untouched)
# ------------------------------------------------------------------ #
organize("/path/to/my/documents", mode="copy")


# ------------------------------------------------------------------ #
# Example 4 — Force re-embed after adding new files
# ------------------------------------------------------------------ #
organize("/path/to/my/documents", force=True)


# ------------------------------------------------------------------ #
# Example 5 — Undo the last move operation
# ------------------------------------------------------------------ #
undo(store_dir="/path/to/my/documents/.semantic_store")


# ------------------------------------------------------------------ #
# Example 6 — Custom store location
# ------------------------------------------------------------------ #
organize(
    folder="/path/to/my/documents",
    store_dir="/custom/store/location",
    mode="copy",
    dry_run=True,
)
