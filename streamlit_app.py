from __future__ import annotations

import os
import tempfile
import streamlit as st

from src.config import load_env, get_settings
from src.observability import init_tracing
from src.openai_client import OpenAIClient
from src.opensearch_service import OpenSearchService
from src.chroma_store import ChromaStore
from src.rag_pipeline import ProdRAG
from src.loaders import load_any

st.set_page_config(page_title="Prod RAG (OpenSearch + Chroma)", layout="wide")

@st.cache_resource
def get_pipeline():
    load_env(".env")
    s = get_settings()
    init_tracing(service_name=s.service_name, otlp_endpoint=s.otlp_endpoint)

    openai = OpenAIClient(api_key=s.openai_api_key)
    os_service = OpenSearchService(host=s.opensearch_host, index=s.opensearch_index, username=s.opensearch_user, password=s.opensearch_pass)
    chroma = ChromaStore(persist_dir=s.chroma_persist_dir, collection=s.chroma_collection)

    rag = ProdRAG(
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
    return rag, s

rag, settings = get_pipeline()
# st.sidebar.write("CHROMA_PERSIST_DIR =", settings.chroma_persist_dir)
# st.sidebar.write("ABS PATH =", os.path.abspath(settings.chroma_persist_dir))

left, right = st.columns([3, 7], gap="large")

with left:
    st.header("📁 Load files")
    st.caption("Upload up to 100 files at a time (pdf, txt, docx, csv, json, html).")

    uploaded = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "docx", "csv", "json", "html", "htm", "ods"],
        accept_multiple_files=True
    )

    if uploaded:
        if len(uploaded) > 100:
            st.error(f"You uploaded {len(uploaded)} files. Please upload up to 100 at a time.")
        else:
            if st.button("🚀 Ingest uploaded files", use_container_width=True):
                ingested_total = 0
                with st.spinner("Ingesting files and building indexes..."):
                    for uf in uploaded:
                        suffix = os.path.splitext(uf.name)[1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uf.getbuffer())
                            tmp_path = tmp.name
                        try:
                            text, fmt = load_any(tmp_path)
                            base_metadata = {"file_name": uf.name}
                            ingested_total += rag.upsert_document(file_name=uf.name, text=text, fmt=fmt, base_metadata=base_metadata)
                        finally:
                            try: os.remove(tmp_path)
                            except Exception: pass

                st.success(f"Ingestion complete. Total chunks added: {ingested_total}")

    st.divider()
    st.subheader("Settings")
    st.write({
        "OpenSearch index": settings.opensearch_index,
        "Chroma dir": settings.chroma_persist_dir,
        "Embed model": settings.openai_embed_model,
        "LLM model": settings.openai_model,
        "Chunk max tokens": settings.chunk_max_tokens,
        "Overlap": settings.chunk_overlap_tokens,
        "Retrieval K": settings.retr_k,
        "Rerank top N": settings.rerank_top_n,
    })

with right:
    st.header("💬 Ask a question")
    query = st.text_area(
        "Your question",
        placeholder="Example: What does the policy say about water damage coverage? Provide clause and cite file.",
        height=120,
    )

    colA, colB = st.columns([1, 1])
    ask = colA.button("Ask", type="primary", use_container_width=True)
    show_evidence = colB.toggle("Show evidence preview", value=True)

    if ask and query.strip():
        with st.spinner("Retrieving, reranking, and generating answer..."):
            res = rag.answer(query)

        st.subheader("Answer")
        st.write(res["answer"])

        st.subheader("How this answer was derived")
        cited_files = []
        for c in res.get("citations", []):
            fn = c.get("file_name")
            if fn and fn not in cited_files:
                cited_files.append(fn)

        if cited_files:
            st.write("**Files referenced:**")
            st.write(cited_files)
        else:
            st.write("No citations available.")

        st.write("**Citations:**")
        st.json(res.get("citations", []), expanded=False)

        if show_evidence:
            st.write("**Evidence preview (top chunks):**")
            for ev in res.get("evidence_preview", []):
                st.markdown(f"**[{ev['cite_id']}] {ev.get('file_name','')}**")
                st.code(ev.get("text",""), language="text")
