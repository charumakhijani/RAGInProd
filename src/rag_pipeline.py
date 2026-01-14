from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

from .caching import RAGCaches
from .chunking import chunk_by_tokens, chunk_header_aware, normalize_text
from .guardrails import mask_pii, enforce_insufficient_evidence
from .observability import span
from .openai_client import OpenAIClient
from .chroma_store import ChromaStore
from .opensearch_service import OpenSearchService

class ProdRAG:
    def __init__(
        self,
        openai: OpenAIClient,
        embed_model: str,
        llm_model: str,
        rerank_model: str,
        os_service: OpenSearchService,
        chroma: ChromaStore,
        caches: Optional[RAGCaches] = None,
        reranker_type: str = "openai",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunk_max_tokens: int = 350,
        chunk_overlap: int = 40,
        retr_k: int = 80,
        rerank_top_n: int = 20,
        max_context_tokens: int = 1200,
        per_chunk_max_tokens: int = 350,
    ):
        self.openai = openai
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.rerank_model = rerank_model
        self.reranker_type = (reranker_type or "openai").lower()
        self.cross_encoder_model = cross_encoder_model
        self._cross_encoder = None  # lazy-loaded
        self.os = os_service
        self.chroma = chroma
        self.caches = caches or RAGCaches()

        self.chunk_max_tokens = chunk_max_tokens
        self.chunk_overlap = chunk_overlap
        self.retr_k = retr_k
        self.rerank_top_n = rerank_top_n
        self.max_context_tokens = max_context_tokens
        self.per_chunk_max_tokens = per_chunk_max_tokens

        probe = self.embed_texts(["dimension probe"])[0]
        self.os.ensure_index(dim=len(probe))

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = [None] * len(texts)  # type: ignore
        to_call, idxs = [], []
        for i, t in enumerate(texts):
            key = self.caches.emb_key(self.embed_model, t)
            if key in self.caches.emb_cache:
                out[i] = self.caches.emb_cache[key]
            else:
                to_call.append(t); idxs.append(i)
        if to_call:
            embs = self.openai.embed(to_call, model=self.embed_model)
            for i, emb in zip(idxs, embs):
                out[i] = emb
                self.caches.emb_cache[self.caches.emb_key(self.embed_model, texts[i])] = emb
        return out

    def chunk_text(self, text: str, fmt: str) -> List[str]:
        text = normalize_text(text)
        if fmt in ("pdf", "docx", "html", "json"):
            return chunk_header_aware(text, max_tokens=self.chunk_max_tokens, overlap=self.chunk_overlap)
        return chunk_by_tokens(text, max_tokens=self.chunk_max_tokens, overlap=self.chunk_overlap)

    def upsert_document(self, file_name: str, text: str, fmt: str, base_metadata: Dict[str, Any]) -> int:
        chunks = self.chunk_text(text, fmt=fmt)
        if not chunks:
            return 0
        chunks = [mask_pii(c) for c in chunks]

        ids = [f"{file_name}::chunk{i}" for i in range(len(chunks))]
        metas = []
        for i in range(len(chunks)):
            m = dict(base_metadata)
            m.update({"file_name": file_name, "format": fmt, "chunk_i": i})
            metas.append(m)

        with span("pipeline.embed_for_upsert", n=len(chunks)):
            embs = self.embed_texts(chunks)

        rows = [{"chunk_id": cid, "file_name": file_name, "text": txt, "metadata": meta, "embedding": emb}
                for cid, txt, meta, emb in zip(ids, chunks, metas, embs)]
        self.os.bulk_upsert(rows)
        self.chroma.upsert(ids=ids, docs=chunks, metas=metas, embeddings=embs)
        return len(chunks)

    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query = mask_pii(query)
        filters_json = json.dumps(filters or {}, sort_keys=True)
        key = self.caches.retr_key(query, filters_json, str(self.retr_k))
        if key in self.caches.retr_cache:
            return self.caches.retr_cache[key]

        with span("pipeline.retrieve"):
            qemb = self.embed_texts([query])[0]
            os_hits = self.os.hybrid_search(query=query, query_emb=qemb, k=self.retr_k, filters=filters)
            chroma_hits = self.chroma.query(query_emb=qemb, k=self.retr_k, where=filters)

            merged: Dict[str, Dict[str, Any]] = {}
            for h in os_hits:
                merged[h["chunk_id"]] = {**h, "score": 0.6 * h["score"]}
            for h in chroma_hits:
                cid = h["chunk_id"]
                if cid in merged:
                    merged[cid]["score"] += 0.4 * h["score"]
                else:
                    merged[cid] = {**h, "file_name": h.get("metadata", {}).get("file_name"), "score": 0.4 * h["score"]}

            cands = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
            self.caches.retr_cache[key] = cands
            return cands

    def _get_cross_encoder(self):
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder
        except Exception as ex:
            raise RuntimeError(
                "CrossEncoder reranker requested but sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            ) from ex
        self._cross_encoder = CrossEncoder(self.cross_encoder_model)
        return self._cross_encoder

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        top = candidates[: self.rerank_top_n]
        ids_joined = ",".join([c["chunk_id"] for c in top])
        model_id = f"{self.reranker_type}:{self.cross_encoder_model if self.reranker_type=='crossencoder' else self.rerank_model}"
        key = self.caches.rerank_key(model_id, query, ids_joined)
        if key in self.caches.rerank_cache:
            return self.caches.rerank_cache[key]

        if self.reranker_type == "crossencoder":
            ce = self._get_cross_encoder()
            pairs = [(query, c["text"][:2000]) for c in top]
            scores = ce.predict(pairs)
            for c, sc in zip(top, scores):
                c["rerank_score"] = float(sc)
            top.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            self.caches.rerank_cache[key] = top
            return top

        # Default: OpenAI LLM-based reranking (reference-free).
        passages = [{"id": c["chunk_id"], "text": c["text"][:1200]} for c in top]
        prompt = (
            "You are a reranking model. Score each passage for answering the question.\n"
            "Return JSON array: [{id, score}] where score is 0..100.\n\n"
            f"QUESTION:\n{query}\n\nPASSAGES:\n{json.dumps(passages, ensure_ascii=False)}"
        )
        raw = self.openai.generate(prompt, model=self.rerank_model)
        try:
            scores_map = {x["id"]: float(x["score"]) for x in json.loads(raw)}
        except Exception:
            scores_map = {p["id"]: 50.0 for p in passages}

        for c in top:
            c["rerank_score"] = scores_map.get(c["chunk_id"], 0.0)
        top.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        self.caches.rerank_cache[key] = top
        return top

    def _pack_context(self, reranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        used = 0
        cite_id = 0
        per_chunk_char_cap = max(300, self.per_chunk_max_tokens * 4)
        max_char_budget = self.max_context_tokens * 4

        for c in reranked:
            t = (c.get("text") or "")[:per_chunk_char_cap]
            if used + len(t) > max_char_budget:
                break
            out.append({
                "cite_id": cite_id,
                "chunk_id": c.get("chunk_id"),
                "file_name": c.get("file_name") or c.get("metadata", {}).get("file_name"),
                "text": t,
            })
            used += len(t)
            cite_id += 1
        return out

    def answer(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with span("pipeline.answer"):
            query = mask_pii(query)
            cands = self.retrieve(query, filters=filters)
            reranked = self.rerank(query, cands)
            evidence = self._pack_context(reranked)

            context = "\n\n".join([f"[{e['cite_id']}] {e['text']}" for e in evidence])
            prompt = (
                "You are an enterprise assistant. Answer ONLY using the provided context.\n"
                "Cite sources like [0], [1]. If insufficient, say so.\n\n"
                f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:"
            )
            ans = self.openai.generate(prompt, model=self.llm_model)

            citations = [{"cite_id": e["cite_id"], "file_name": e.get("file_name"), "chunk_id": e["chunk_id"]} for e in evidence]
            ans = enforce_insufficient_evidence(ans, citations, min_cites=1)

            return {
                "answer": ans,
                "citations": citations,
                "evidence_preview": [{"cite_id": e["cite_id"], "file_name": e.get("file_name"), "text": e["text"][:400]} for e in evidence],
            }


    def ask(self, question: str):
        """Alias for answer()."""
        return self.answer(question)
