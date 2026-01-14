from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class EvalExample:
    id: str
    query: str
    # Backward-compatible: file names are still supported
    relevant_sources: List[str]
    # More robust identifiers
    relevant_doc_ids: List[str] | None = None
    # Optional reference evidence text (e.g., clause snippet)
    reference_text: Optional[str] = None
    expected_answer: Optional[str] = None
    tags: Dict[str, Any] | None = None

def load_jsonl(path: str) -> List[EvalExample]:
    out: List[EvalExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(EvalExample(
                id=str(obj.get("id", f"line{line_no}")),
                query=str(obj["query"]),
                relevant_sources=list(obj.get("relevant_sources", [])),
                relevant_doc_ids=list(obj.get("relevant_doc_ids", [])) or None,
                reference_text=obj.get("reference_text"),
                expected_answer=obj.get("expected_answer"),
                tags=obj.get("tags"),
            ))
    return out
