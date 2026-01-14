from __future__ import annotations
from typing import List
from openai import OpenAI
from .observability import span

class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def embed(self, texts: List[str], model: str) -> List[List[float]]:
        with span("openai.embeddings", n=len(texts), model=model):
            resp = self.client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    def generate(self, prompt: str, model: str) -> str:
        with span("openai.responses", model=model):
            resp = self.client.responses.create(model=model, input=prompt)
        return resp.output_text
