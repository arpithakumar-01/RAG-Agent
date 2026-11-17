# Enterprise RAG Agents

A multi-agent Retrieval-Augmented Generation (RAG) assistant that answers questions over internal
company documents with **citations**, **self-verification**, and **per-request tracing** of tokens,
latency and cost. Built on the Gemini API and designed to deploy on Google Cloud (Cloud Run).

The bundled sample corpus is a fictional company handbook (security, incident response, leave,
engineering standards, onboarding, expenses) in `data/docs/`. Replace it with your own `.md`/`.txt` files.

## Architecture

```
question
   │
   ▼
Planner agent ──► splits multi-part questions into ≤3 sub-queries (JSON)
   │
   ▼
Retriever agent (ReAct-style) ──► hybrid search: dense embeddings + BM25, fused with RRF
   │   Thought: is the best match weak?  Action: rewrite query  Observation: re-search
   ▼
Answerer agent ──► answers ONLY from numbered passages, cites [n], abstains if unsupported
   │
   ▼
Critic agent (self-reflection) ──► checks every claim against its cited passage;
   │       if unsupported, feedback goes back to the Answerer (max MAX_REVISIONS loops)
   ▼
answer + citations + verified flag + trace (tokens in/out, tokens/sec, est. cost, per-span latency)
```

Key files: `app/ingest.py` (chunking + indexing), `app/vectorstore.py` (hybrid retrieval),
`app/agents.py` (the four agents), `app/pipeline.py` (orchestration), `app/tracing.py` (LLM metrics),
`app/api.py` (FastAPI REST), `eval/run_eval.py` (evaluation harness).

## Quick start

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key        # without a key the app runs in offline MOCK mode
python -m app.cli ingest              # build the vector index from data/docs
python -m app.cli ask "How long must a password be?"
uvicorn app.api:app --reload          # POST /ask {"question": "..."}  |  GET /health
python -m eval.run_eval               # evaluation report
pytest -q                             # tests
```
## Ideas to extend

- Swap the plain-Python orchestration for Google ADK or LangGraph (agents already have clean interfaces).
- Replace the in-memory store with Vertex AI Vector Search, AlloyDB or ChromaDB.
- Add document-level access control and PII redaction before indexing.
- Grow `eval/eval_set.json` and add LLM-as-judge faithfulness scoring.
