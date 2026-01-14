from __future__ import annotations
from cachetools import TTLCache
import hashlib

class RAGCaches:
    def __init__(
        self,
        embed_max=50_000, embed_ttl=3600,
        retr_max=50_000, retr_ttl=300,
        rerank_max=50_000, rerank_ttl=300,
    ):
        self.emb_cache = TTLCache(maxsize=embed_max, ttl=embed_ttl)
        self.retr_cache = TTLCache(maxsize=retr_max, ttl=retr_ttl)
        self.rerank_cache = TTLCache(maxsize=rerank_max, ttl=rerank_ttl)

    @staticmethod
    def _key(*parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8"))
            h.update(b"|")
        return h.hexdigest()

    def emb_key(self, model: str, text: str) -> str:
        return self._key("emb", model, text)

    def retr_key(self, query: str, filters_json: str, k: str) -> str:
        return self._key("retr", query, filters_json, k)

    def rerank_key(self, model: str, query: str, ids_joined: str) -> str:
        return self._key("rerank", model, query, ids_joined)
