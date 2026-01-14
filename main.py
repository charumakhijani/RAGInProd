from __future__ import annotations
from src.config import load_env, get_settings
from src.observability import init_tracing
from src.openai_client import OpenAIClient
from src.opensearch_service import OpenSearchService
from src.chroma_store import ChromaStore
from src.rag_pipeline import ProdRAG

def build_pipeline():
    load_env(".env")
    s = get_settings()
    init_tracing(service_name=s.service_name, otlp_endpoint=s.otlp_endpoint)

    openai = OpenAIClient(api_key=s.openai_api_key)
    os_service = OpenSearchService(host=s.opensearch_host, index=s.opensearch_index, username=s.opensearch_user, password=s.opensearch_pass)
    chroma = ChromaStore(persist_dir=s.chroma_persist_dir, collection=s.chroma_collection)

    return ProdRAG(
        openai=openai,
        embed_model=s.openai_embed_model,
        llm_model=s.openai_model,
        rerank_model=s.openai_rerank_model,
        reranker_type=getattr(s, 'reranker_type', 'openai'),
        cross_encoder_model=getattr(s, 'cross_encoder_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
        os_service=os_service,
        chroma=chroma,
        chunk_max_tokens=s.chunk_max_tokens,
        chunk_overlap=s.chunk_overlap_tokens,
        retr_k=s.retr_k,
        rerank_top_n=s.rerank_top_n,
        max_context_tokens=s.max_context_tokens,
        per_chunk_max_tokens=s.per_chunk_max_tokens,
    )

if __name__ == "__main__":
    _ = build_pipeline()
    print("✅ Pipeline ready. Run: streamlit run streamlit_app.py")
