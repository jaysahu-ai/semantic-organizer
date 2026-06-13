# semantic-organizer

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Offline](https://img.shields.io/badge/100%25-Offline-teal?style=flat-square)
![Version](https://img.shields.io/badge/version-0.1.0-gray?style=flat-square)
![PyPI](https://img.shields.io/pypi/v/semantic-organizer?style=flat-square&color=blue)
![Downloads](https://img.shields.io/pypi/dm/semantic-organizer?style=flat-square&color=green)

> Automatically organize your local documents into meaningful folders — fully offline, no LLMs, no API keys, no cloud.

```bash
pip install semantic-organizer
```

```python
from semantic_organizer import organize

organize("/path/to/my/documents")
```

# Windows
organize("C:\\Users\\yourname\\Documents")

# or using raw string
organize(r"C:\Users\yourname\Documents")

---

## What it does

Point it at any folder. It reads your `.txt`, `.pdf`, and `.docx` files, understands what they mean, groups them by topic, and moves them into labeled subfolders — automatically.

**Before**
```
documents/
├── q3_report.pdf
├── lecture_notes.txt
├── sprint_retro.docx
├── research_paper.pdf
├── roadmap.txt
└── meeting_notes.docx
```

**After**
```
documents/
├── Financial Reports/
│   └── q3_report.pdf
├── Machine Learning/
│   ├── lecture_notes.txt
│   └── research_paper.pdf
└── Project Management/
    ├── sprint_retro.docx
    ├── roadmap.txt
    └── meeting_notes.docx
```

---

## How it works

```
Documents → Extract text → Embed with all-MiniLM-L6-v2 → KMeans clustering → KeyBERT labels → Organize
```

1. **Extracts** text from `.txt`, `.pdf`, and `.docx` files
2. **Embeds** each document using `all-MiniLM-L6-v2` — a 80MB model that runs fully offline after first download
3. **Clusters** documents by semantic similarity using KMeans + Silhouette Analysis to auto-detect the optimal number of groups
4. **Labels** each cluster with descriptive keywords using KeyBERT
5. **Organizes** files into labeled subfolders

Embeddings are cached locally after the first run — subsequent runs on the same folder are instant.

---

## Installation

```bash
pip install semantic-organizer
```

Requires Python 3.10+. The embedding model (~80MB) is downloaded on first use and cached locally.

---

## Usage

### Basic

```python
from semantic_organizer import organize

organize("/path/to/my/documents")
```

### Dry run — preview before committing

Always a good idea on first use:

```python
clusters = organize("/path/to/my/documents", dry_run=True)

# Inspect the proposed structure
for cluster in clusters:
    print(cluster["label"])
    for doc in cluster["documents"]:
        print(f"  {doc['filename']}")
```

### Copy mode — originals untouched

```python
organize("/path/to/my/documents", mode="copy")
```

### Undo last operation

```python
from semantic_organizer import undo

undo(store_dir="/path/to/my/documents/.semantic_store")
```

### All options

```python
organize(
    folder="/path/to/my/documents",

    # "move" (default) — moves original files into labeled subfolders
    # "copy"           — copies files, originals stay untouched
    mode="copy",

    # Where to persist embeddings between runs
    # Default: <folder>/.semantic_store
    store_dir="/custom/store/path",

    # Re-embed all documents even if a store already exists
    # Use this after adding or removing files from the folder
    force=False,

    # Preview without moving or copying anything
    dry_run=True,
)
```

---

## Return value

`organize()` always returns the cluster list — whether or not `dry_run` is set:

```python
[
    {
        "cluster_id": 0,
        "label": "Machine Learning",
        "documents": [
            {"filename": "lecture_notes.txt", "path": "/abs/path/lecture_notes.txt"},
            {"filename": "research_paper.pdf", "path": "/abs/path/research_paper.pdf"},
        ]
    },
    ...
]
```

---

## Privacy

Everything runs locally on your machine.

- No files, text, or embeddings are ever sent to a server
- The embedding model (`all-MiniLM-L6-v2`) is downloaded once from HuggingFace and cached locally
- All subsequent runs are completely offline

---

## Architecture

```
semantic_organizer/
├── extractor.py    — text extraction from .txt, .pdf, .docx
├── embedder.py     — sentence embeddings via all-MiniLM-L6-v2
├── store.py        — persist embeddings as .npy + metadata as .json
├── clusterer.py    — KMeans + Silhouette Analysis
├── labeler.py      — KeyBERT keyword extraction
└── controller.py   — move/copy files + manifest-based undo
```

---

## Known limitations

- **Short documents** (under 50 chars) are skipped — too little text to embed reliably
- **Non-English documents** may cluster poorly — the default model is optimized for English
- **Large folders** (500+ files) will be slow on the first run — embedding is the bottleneck. Subsequent runs use the cached store
- **Highly uniform folders** (all documents on the same topic) may produce less meaningful clusters

---

## Development

```bash
git clone git clone https://github.com/jaysahu-ai/semantic-organizer
cd semantic-organizer

python3 -m venv venv
source venv/bin/activate

pip install -e ".[dev]"

pytest tests/ -v
```

---

## License

MIT
