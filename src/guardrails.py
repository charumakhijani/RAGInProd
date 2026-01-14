from __future__ import annotations
import re
from typing import Any, Dict, List

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"\b(\+?\d{1,2}[\s-]?)?(\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b")
SSN_RE   = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

def mask_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = SSN_RE.sub("[SSN]", text)
    return text

def enforce_insufficient_evidence(answer: str, citations: List[Dict[str, Any]], min_cites: int = 1) -> str:
    if len(citations) < min_cites:
        return "I don’t have enough information in the provided sources to answer that confidently."
    return answer
