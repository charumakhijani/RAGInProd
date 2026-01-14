from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import chromadb


class ChromaStore:
    def __init__(self, persist_dir: str, collection: str):
        persist_dir = os.path.abspath(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)

        # Use PersistentClient to avoid sqlite schema issues
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.col = self.client.get_or_create_collection(collection)

    def upsert(
        self,
        ids: List[str],
        docs: List[str],
        metas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ):
        self.col.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings,
        )

    def query(
        self,
        query_emb: List[float],
        k: int,
        where: Optional[Dict[str, Any]] = None,
    ):
        res = self.col.query(
            query_embeddings=[query_emb],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        out = []
        for cid, txt, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            score = float(1.0 / (1.0 + dist))
            out.append(
                {
                    "chunk_id": cid,
                    "text": txt,
                    "metadata": meta,
                    "score": score,
                }
            )

        return out
