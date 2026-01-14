from __future__ import annotations
from typing import Any, Dict, List, Optional
from opensearchpy import OpenSearch
from .observability import span

class OpenSearchService:
    def __init__(self, host: str, index: str, username: str | None = None, password: str | None = None):
        auth = (username, password) if username and password else None
        self.client = OpenSearch(hosts=[host], http_auth=auth)
        self.index = index

    def ensure_index(self, dim: int):
        if self.client.indices.exists(index=self.index):
            return
        body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "file_name": {"type": "keyword"},
                    "text": {"type": "text"},
                    "metadata": {"type": "object", "enabled": True},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": {"name": "hnsw", "engine": "nmslib", "space_type": "cosinesimil"}
                    }
                }
            }
        }
        self.client.indices.create(index=self.index, body=body)

    def bulk_upsert(self, rows: List[Dict[str, Any]]):
        with span("opensearch.bulk_upsert", n=len(rows)):
            bulk = []
            for r in rows:
                bulk.append({"index": {"_index": self.index, "_id": r["chunk_id"]}})
                bulk.append(r)
            self.client.bulk(body=bulk, refresh=True)

    def hybrid_search(self, query: str, query_emb: List[float], k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with span("opensearch.hybrid_search", k=k):
            flt = []
            if filters:
                for key, val in filters.items():
                    flt.append({"term": {f"metadata.{key}": val}})
            body = {
                "size": k,
                "query": {
                    "bool": {
                        "filter": flt,
                        "should": [
                            {"match": {"text": {"query": query, "boost": 1.0}}},
                            {"knn": {"embedding": {"vector": query_emb, "k": k, "boost": 1.0}}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
            }
            res = self.client.search(index=self.index, body=body)
            hits = res.get("hits", {}).get("hits", [])
            out = []
            for h in hits:
                src = h["_source"]
                out.append({
                    "chunk_id": src["chunk_id"],
                    "file_name": src.get("file_name"),
                    "text": src["text"],
                    "metadata": src.get("metadata", {}),
                    "score": float(h.get("_score", 0.0)),
                })
            return out
