from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.config import load_env, get_settings
from src.observability import init_tracing
from src.openai_client import OpenAIClient
from src.opensearch_service import OpenSearchService
from src.chroma_store import ChromaStore
from src.rag_pipeline import ProdRAG

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from eval.dataset import load_jsonl


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


def to_ragas_dataset(rag: ProdRAG, examples, k: int) -> Dataset:
    records: List[Dict[str, Any]] = []
    for ex in examples:
        retrieved = rag.retrieve(ex.query)
        contexts = [r.get("text", "") for r in retrieved[:k] if r.get("text")]

        res = rag.answer(ex.query)
        records.append(
            {
                "question": ex.query,
                "answer": res.get("answer", ""),
                "contexts": contexts,
                # If you provide expected_answer in JSONL, context_recall becomes meaningful.
                "ground_truth": ex.expected_answer or "",
            }
        )
    return Dataset.from_list(records)


def main():
    ap = argparse.ArgumentParser(description="RAGAS evaluation runner for the local RAG pipeline")
    ap.add_argument("--dataset", default="eval/sample_golden.jsonl", help="JSONL with {query,...}. Uses 'query' field.")
    ap.add_argument("--k", type=int, default=8, help="Top-k contexts passed to RAGAS")
    ap.add_argument("--out", default="eval/ragas_report.csv", help="Output CSV path")
    ap.add_argument("--metrics", default="faithfulness,answer_relevancy,context_precision,context_recall",
                    help="Comma-separated metric names")
    args = ap.parse_args()

    examples = load_jsonl(args.dataset)

    rag, s = build_pipeline()

    eval_llm = LangchainLLMWrapper(ChatOpenAI(model=s.openai_model, temperature=0))
    eval_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=s.openai_embed_model))

    ds = to_ragas_dataset(rag, examples, k=args.k)

    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    metrics = [metric_map[m.strip()] for m in args.metrics.split(",") if m.strip() in metric_map]

    result = evaluate(ds, metrics=metrics, llm=eval_llm, embeddings=eval_emb)
    df = result.to_pandas()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print("=== RAGAS results (aggregate) ===")
    print(df.mean(numeric_only=True))
    print(f"\nWrote report: {args.out}")


if __name__ == "__main__":
    main()
