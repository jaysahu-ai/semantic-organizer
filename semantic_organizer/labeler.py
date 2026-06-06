"""
labeler.py
----------
Generates clean, generalized folder labels for each cluster using KeyBERT.

Strategy (in priority order):
  1. Filename pattern detection — if filenames share a common prefix
     like receipt_, travel_, code_ that's the most reliable signal
  2. Per-document keyword voting — extract keywords from each doc
     individually and pick the most common across the cluster
  3. Fallback to "Uncategorized" if both fail
"""

from collections import Counter
from keybert import KeyBERT

_kw_model = None

NOISE_WORDS = {
    "total", "payment", "paid", "amount", "price", "cost", "rate",
    "date", "time", "item", "items", "file", "files", "document",
    "pdf", "docx", "txt", "page", "pages", "section", "update",
    "week", "day", "days", "month", "year", "upi", "cash", "card",
    "print", "return", "import", "def", "class", "self", "none",
    "true", "false", "null", "count", "group", "select", "from",
    "where", "order", "limit", "done", "pending", "completed",
    "highlight", "highlights", "risk", "risks", "plan", "sprint",
    "status", "weekly", "build", "building",
}

# Filename prefix → clean human-readable label
# If majority of files in a cluster share a prefix, use this label directly
FILENAME_HINTS = {
    "receipt":  "Receipts",
    "invoice":  "Invoices",
    "travel":   "Travel Plans",
    "trip":     "Travel Plans",
    "project":  "Project Status",
    "status":   "Project Status",
    "code":     "Code Snippets",
    "script":   "Code Snippets",
    "chat":     "Conversations",
    "message":  "Conversations",
    "note":     "Notes",
    "report":   "Reports",
}


def get_kw_model() -> KeyBERT:
    global _kw_model
    if _kw_model is None:
        print("[labeler] Loading KeyBERT...")
        _kw_model = KeyBERT(model="all-MiniLM-L6-v2")
        print("[labeler] KeyBERT ready.")
    return _kw_model


def _clean_keyword(keyword: str) -> str:
    return keyword.replace("_", " ").replace("-", " ").strip().lower()


def _is_noise(keyword: str) -> bool:
    words = keyword.lower().split()
    return all(w in NOISE_WORDS for w in words)


def _detect_from_filenames(documents: list[dict]) -> str | None:
    """
    Check if a majority of filenames share a known prefix.
    Returns a clean label if found, None otherwise.

    e.g. [receipt_fuel.txt, receipt_grocery.txt, receipt_pharmacy.txt]
         → 3/3 match "receipt" → "Receipts"
    """
    total = len(documents)
    if total == 0:
        return None

    for prefix, label in FILENAME_HINTS.items():
        matches = sum(
            1 for doc in documents
            if doc["filename"].lower().startswith(prefix)
        )
        # More than half the files match this prefix → use it
        if matches / total > 0.5:
            return label

    return None


def _extract_keywords_from_text(text: str, top_n: int = 5) -> list[str]:
    model = get_kw_model()
    if not text.strip():
        return []

    results = model.extract_keywords(
        text[:500],
        keyphrase_ngram_range=(1, 2),
        top_n=top_n,
        use_mmr=True,
        diversity=0.6,
        stop_words="english",
    )
    cleaned = [_clean_keyword(kw) for kw, _ in results]
    return [kw for kw in cleaned if not _is_noise(kw)]


def generate_cluster_label(
    documents: list[dict],
    full_texts: dict[str, str],
) -> str:
    """
    Generate a label using filename hints first, keyword voting as fallback.
    """
    # Priority 1 — filename pattern detection (most reliable)
    hint = _detect_from_filenames(documents)
    if hint:
        return hint

    # Priority 2 — per-document keyword voting
    all_keywords = []
    for doc in documents:
        text = full_texts.get(doc["path"], "")
        keywords = _extract_keywords_from_text(text)
        all_keywords.extend(keywords)

    if not all_keywords:
        return "Uncategorized"

    counter = Counter(all_keywords)
    best = counter.most_common(1)[0][0]
    return best.title()


def label_clusters(
    clusters: list[dict],
    full_texts: dict[str, str],
    top_n: int = 3,
) -> list[dict]:
    print(f"[labeler] Generating labels for {len(clusters)} cluster(s)...")

    for cluster in clusters:
        label = generate_cluster_label(cluster["documents"], full_texts)
        cluster["label"] = label
        print(f"[labeler] Cluster {cluster['cluster_id']} → '{label}' ({len(cluster['documents'])} docs)")

    return clusters