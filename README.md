# Prod RAG Project (OpenSearch BM25 + k-NN + Chroma) + Streamlit UI

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configure environment
Copy `.env.example` to `.env` and fill values:
```bash
cp .env.example .env
```

## Run app
```bash
streamlit run streamlit_app.py
```

## UI
- Left pane (~30%): upload up to 100 files, ingest into OpenSearch+Chroma
- Right pane (~70%): ask question; see answer + files referenced + citations

## Evaluation

Create a small "golden set" of questions + expected evidence in `eval/*.jsonl` (see `eval/sample_golden.jsonl`).

Run retrieval-only evaluation:
```bash
python -m eval.runner --dataset eval/sample_golden.jsonl --k 20 --no-answer
```

Run retrieval + answer evaluation (adds rule-based citation checks):
```bash
python -m eval.runner --dataset eval/sample_golden.jsonl --k 20
```

Outputs a CSV report at `eval/report.csv` with per-question metrics (P@k, Recall@k, MRR@k, citation presence/hit).

## RAGAS Evaluation

This project includes a reference-free RAGAS evaluation runner at `eval/ragas_runner.py`.
It evaluates your pipeline using these common RAG metrics:
- faithfulness
- answer_relevancy
- context_precision
- context_recall

Install deps:
```bash
pip install -r requirements.txt
```

Run:
```bash
python -m eval.ragas_runner --dataset eval/sample_golden.jsonl --k 8
```

Output:
- `eval/ragas_report.csv` (per-question scores)

### Reranker

Set `RERANKER_TYPE=openai` (default) to use the OpenAI-based JSON scoring reranker.
Set `RERANKER_TYPE=crossencoder` to use a local SentenceTransformers CrossEncoder reranker.

Optional:
- `CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`
