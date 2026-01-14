from __future__ import annotations
import re
from typing import List
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")

def normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def chunk_by_tokens(text: str, max_tokens: int, overlap: int) -> List[str]:
    ids = ENC.encode(text)
    out = []
    start = 0
    while start < len(ids):
        end = min(len(ids), start + max_tokens)
        out.append(ENC.decode(ids[start:end]))
        if end >= len(ids):
            break
        start = max(0, end - overlap)
    return [c.strip() for c in out if c.strip()]

def chunk_header_aware(text: str, max_tokens: int, overlap: int) -> List[str]:
    parts, cur = [], []
    for line in text.splitlines():
        if line.strip().startswith(("#", "##", "###")) or re.match(r"^[A-Z][A-Z0-9 \-]{5,}$", line.strip()):
            if cur:
                parts.append("\n".join(cur).strip()); cur = []
        cur.append(line)
    if cur:
        parts.append("\n".join(cur).strip())

    chunks: List[str] = []
    for p in parts:
        if len(ENC.encode(p)) <= max_tokens:
            chunks.append(p)
        else:
            chunks.extend(chunk_by_tokens(p, max_tokens=max_tokens, overlap=overlap))
    return [c.strip() for c in chunks if c.strip()]
