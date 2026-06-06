"""
extractor.py
------------
Extracts plain text from .txt, .pdf, and .docx files.

Supported formats (per design doc):
  - .txt  → read directly
  - .pdf  → PyMuPDF (fitz)
  - .docx → python-docx
"""

from pathlib import Path


def extract_txt(file_path: Path) -> str:
    """Read a plain text file."""
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception as e:
        print(f"[extractor] Failed to read TXT {file_path.name}: {e}")
        return ""


def extract_pdf(file_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(file_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages).strip()
    except Exception as e:
        print(f"[extractor] Failed to read PDF {file_path.name}: {e}")
        return ""


def extract_docx(file_path: Path) -> str:
    """Extract text from a .docx file using python-docx."""
    try:
        from docx import Document
        doc = Document(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        print(f"[extractor] Failed to read DOCX {file_path.name}: {e}")
        return ""


def extract_text(file_path: Path) -> str:
    """
    Dispatch to the correct extractor based on file extension.
    Returns an empty string if the format is unsupported or extraction fails.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return extract_txt(file_path)
    elif suffix == ".pdf":
        return extract_pdf(file_path)
    elif suffix == ".docx":
        return extract_docx(file_path)
    else:
        print(f"[extractor] Unsupported format: {file_path.name}")
        return ""


def extract_folder(folder_path: Path, min_chars: int = 50) -> list[dict]:
    """
    Walk a folder and extract text from all supported files.

    Args:
        folder_path: Directory to scan.
        min_chars:   Minimum character count to keep a document.
                     Files below this threshold are skipped (too short to embed reliably).
                     See design doc — Known Limitation: short documents produce poor embeddings.

    Returns:
        List of dicts: [{"path": Path, "text": str}, ...]
    """
    supported = {".txt", ".pdf", ".docx"}
    results = []

    files = [f for f in folder_path.rglob("*") if f.is_file() and f.suffix.lower() in supported]

    if not files:
        print(f"[extractor] No supported files found in {folder_path}")
        return results

    print(f"[extractor] Found {len(files)} supported file(s). Extracting...")

    for file in files:
        text = extract_text(file)
        if len(text) < min_chars:
            print(f"[extractor] Skipping {file.name} — too short ({len(text)} chars, min={min_chars})")
            continue
        results.append({"path": file, "text": text})
        print(f"[extractor] OK  {file.name} ({len(text)} chars)")

    print(f"[extractor] Extracted {len(results)}/{len(files)} files successfully.")
    return results
