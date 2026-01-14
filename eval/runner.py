from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List, Optional

from eval.dataset import load_jsonl
from eval.metrics import precision_at_k, recall_at_k, mrr_at_k, dedupe_preserve, cosine

from src.config import load_env, get_settings
from src.observability import init_tracing
from src.openai_client import OpenAIClient
from src.opensearch_service import OpenSearchService
from src.chroma_store import ChromaStore
from src.rag_pipeline import ProdRAG


def build_pipeline() -> tuple[ProdRAG, Any]:
    load_env(".env")
    s = get_settings()
    init_tracing(service_name=s.service_name, otlp_endpoint=s.otlp_endpoint)

    openai = OpenAIClient(api_key=s.openai_api_key)
    os_service = OpenSearchService(
        host=s.opensearch_host,
        index=s.opensearch_index,
        username=s.opensearch_user,
        password=s.opensearch_pass,
    )
    chroma = ChromaStore(persist_dir=s.chroma_persist_dir, collection=s.chroma_collection)

    rag = ProdRAG(
        openai=openai,
        embed_model=s.openai_embed_model,
        llm_model=s.openai_model,
        rerank_model=s.openai_rerank_model,
        os_service=os_service,
        chroma=chroma,
        chunk_max_tokens=s.chunk_max_tokens,
        chunk_overlap=s.chunk_overlap_tokens,
        retr_k=s.retr_k,
        rerank_top_n=s.rerank_top_n,
        max_context_tokens=s.max_context_tokens,
        per_chunk_max_tokens=s.per_chunk_max_tokens,
    )
    return rag, s


def extract_key(r: Dict[str, Any], match: str) -> str:
    meta = r.get("metadata", {}) or {}
    if match == "chunk_id":
        return str(r.get("chunk_id", ""))
    if match == "doc_id":
        return str(meta.get("document_id") or "")
    # default: file_name
    return str(r.get("file_name") or meta.get("file_name") or "")


def semantic_relevance(
    rag: ProdRAG,
    retrieved: List[Dict[str, Any]],
    reference_text: str,
    k: int,
    threshold: float,
) -> List[bool]:
    # Embed reference once + embed each retrieved context (top-k)
    ref_emb = rag.embed_texts([reference_text])[0]
    ctx_texts = [r.get("text", "") for r in retrieved[:k]]
    ctx_embs = rag.embed_texts(ctx_texts) if ctx_texts else []
    flags: List[bool] = []
    for emb in ctx_embs:
        flags.append(cosine(ref_emb, emb) >= threshold)
    # pad if needed
    while len(flags) < min(k, len(retrieved)):
        flags.append(False)
    return flags


def clause_relevance(retrieved: List[Dict[str, Any]], reference_text: str, k: int) -> List[bool]:
    ref = (reference_text or "").strip().lower()
    flags: List[bool] = []
    for r in retrieved[:k]:
        txt = (r.get("text") or "").lower()
        flags.append(bool(ref) and ref in txt)
    return flags


def main():
    ap = argparse.ArgumentParser(description="Evaluate retrieval + answer grounding for the local RAG pipeline")
    ap.add_argument("--dataset", default="eval/sample_golden.jsonl", help="Path to JSONL dataset")
    ap.add_argument("--k", type=int, default=20, help="Top-k for retrieval metrics")
    ap.add_argument(
        "--match",
        choices=["file_name", "chunk_id", "doc_id", "semantic", "clause"],
        default="file_name",
        help="How to determine relevance. semantic/clause use reference_text in dataset.",
    )
    ap.add_argument("--semantic-threshold", type=float, default=0.80, help="Cosine threshold for semantic match")
    ap.add_argument("--out", default="eval/report.csv", help="CSV output path")
    ap.add_argument("--no-answer", action="store_true", help="Only evaluate retrieval; skip answer generation")
    args = ap.parse_args()

    rag, _settings = build_pipeline()
    examples = load_jsonl(args.dataset)

    rows: List[Dict[str, Any]] = []
    p_list: List[float] = []
    r_list: List[float] = []
    mrr_list: List[float] = []
    cite_hit_list: List[float] = []
    cite_present_list: List[float] = []

    for ex in examples:
        retrieved = rag.retrieve(ex.query)

        # Determine relevance flags for top-k
        if args.match == "semantic":
            if not ex.reference_text:
                raise ValueError("match=semantic requires 'reference_text' in the JSONL example")
            rels = semantic_relevance(rag, retrieved, ex.reference_text, args.k, args.semantic_threshold)
            keys = [extract_key(r, "file_name") for r in retrieved[:args.k]]
            gt_str = ex.reference_text[:200].replace("\n", " ")
        elif args.match == "clause":
            if not ex.reference_text:
                raise ValueError("match=clause requires 'reference_text' in the JSONL example")
            rels = clause_relevance(retrieved, ex.reference_text, args.k)
            keys = [extract_key(r, "file_name") for r in retrieved[:args.k]]
            gt_str = ex.reference_text[:200].replace("\n", " ")
        else:
            # file_name/chunk_id/doc_id
            gt_set = set(ex.relevant_sources or [])
            if args.match == "doc_id":
                gt_set = set(ex.relevant_doc_ids or [])
            rels = []
            keys = []
            for r in retrieved[:args.k]:
                key = extract_key(r, args.match)
                keys.append(key)
                rels.append(bool(key) and (key in gt_set))
            gt_str = "|".join(sorted(gt_set))

        p = precision_at_k(rels, args.k)
        r = recall_at_k(rels, args.k)
        mrr = mrr_at_k(rels, args.k)
        p_list.append(p); r_list.append(r); mrr_list.append(mrr)

        answer = ""
        citations = []
        cited_files: List[str] = []
        cite_present = False
        cite_hit = False

        if not args.no_answer:
            res = rag.answer(ex.query)
            answer = res.get("answer", "")
            citations = res.get("citations", []) or []
            cite_present = len(citations) > 0
            cited_files = dedupe_preserve([c.get("file_name") for c in citations if c.get("file_name")])

            # Citation hit logic aligns with match mode
            if args.match == "chunk_id":
                gt = set(ex.relevant_sources or [])  # allow chunk ids in relevant_sources for compatibility
                cite_hit = any((c.get("chunk_id") in gt) for c in citations)
            elif args.match == "doc_id":
                gt = set(ex.relevant_doc_ids or [])
                cite_hit = any(((c.get("metadata", {}) or {}).get("document_id") in gt) for c in citations)
            elif args.match in ("semantic", "clause"):
                # If you use semantic/clause, treat citation hit as any citation file present (best-effort).
                cite_hit = cite_present
            else:
                gt = set(ex.relevant_sources or [])
                cite_hit = any((c.get("file_name") in gt) for c in citations)

        cite_present_list.append(1.0 if cite_present else 0.0)
        cite_hit_list.append(1.0 if cite_hit else 0.0)

        rows.append({
            "id": ex.id,
            "query": ex.query,
            "k": args.k,
            "match": args.match,
            "precision_at_k": round(p, 4),
            "recall_at_k": round(r, 4),
            "mrr_at_k": round(mrr, 4),
            "ground_truth": gt_str,
            "topk_retrieved_keys": "|".join(keys[: args.k]),
            "citations_present": int(cite_present),
            "citation_hit": int(cite_hit),
            "cited_files": "|".join(cited_files),
            "answer": answer.replace("\n", " ")[:2000],
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    def avg(xs: List[float]) -> float:
        return sum(xs)/len(xs) if xs else 0.0

    print("=== Retrieval metrics ===")
    print(f"N={len(rows)}  k={args.k}  match={args.match}")
    print(f"Precision@{args.k}: {avg(p_list):.3f}")
    print(f"Recall@{args.k}:    {avg(r_list):.3f}")
    print(f"MRR@{args.k}:       {avg(mrr_list):.3f}")

    if not args.no_answer:
        print("\n=== Answer metrics (rule-based) ===")
        print(f"Citations present rate: {avg(cite_present_list):.3f}")
        print(f"Citation hit rate:      {avg(cite_hit_list):.3f}")

    print(f"\nWrote report: {args.out}")


if __name__ == "__main__":
    main()
