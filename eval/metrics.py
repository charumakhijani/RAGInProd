from __future__ import annotations
from typing import Dict, List, Tuple, Any
import math

def precision_at_k(rels: List[bool], k: int) -> float:
    k = min(k, len(rels))
    if k == 0:
        return 0.0
    return sum(1 for x in rels[:k] if x) / float(k)

def recall_at_k(rels: List[bool], k: int) -> float:
    k = min(k, len(rels))
    if k == 0:
        return 0.0
    return 1.0 if any(rels[:k]) else 0.0

def mrr_at_k(rels: List[bool], k: int) -> float:
    k = min(k, len(rels))
    for i in range(k):
        if rels[i]:
            return 1.0 / float(i + 1)
    return 0.0

def dedupe_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
