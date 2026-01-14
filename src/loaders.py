from __future__ import annotations
import os, json
from typing import Tuple
import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document as DocxDocument

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for i, p in enumerate(reader.pages):
        pages.append(f"\n\n--- Page {i+1} ---\n" + (p.extract_text() or ""))
    return "\n".join(pages)

def load_docx(path: str) -> str:
    doc = DocxDocument(path)
    lines = []
    for para in doc.paragraphs:
        t = (para.text or "").strip()
        if t:
            lines.append(t)
    return "\n\n".join(lines)

def load_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return soup.get_text("\n") or ""

def load_csv(path: str) -> str:
    df = pd.read_csv(path)
    rows = []
    for idx, row in df.iterrows():
        rows.append("ROW %s:\n" % idx + "\n".join([f"{c}: {row[c]}" for c in df.columns]))
    return "\n\n".join(rows)

def load_json(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        obj = json.load(f)
    return json.dumps(obj, indent=2, ensure_ascii=False)

def load_ods(path: str) -> str:
    xls = pd.ExcelFile(path, engine="odf")
    parts = []
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        parts.append(f"=== SHEET: {sheet_name} ===")
        for idx, row in df.iterrows():
            parts.append("ROW %s:\n" % idx + "\n".join([f"{c}: {row[c]}" for c in df.columns]))
        parts.append("")  # spacer
    return "\n\n".join(parts)

def load_any(path: str) -> Tuple[str, str]:
    ext = os.path.splitext(path.lower())[1]
    if ext == ".txt": return load_txt(path), "txt"
    if ext == ".pdf": return load_pdf(path), "pdf"
    if ext == ".docx": return load_docx(path), "docx"
    if ext in (".html", ".htm"): return load_html(path), "html"
    if ext == ".csv": return load_csv(path), "csv"
    if ext == ".json": return load_json(path), "json"
    if ext == ".ods": return load_ods(path), "ods"
    raise ValueError(f"Unsupported file extension: {ext}")
