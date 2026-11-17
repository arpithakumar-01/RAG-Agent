# Enterprise RAG Agents

A multi-agent Retrieval-Augmented Generation (RAG) assistant that answers questions over company documents with citations, self-verification, and per-request tracing. It uses Gemini for generation and embeddings when a Gemini API key is configured; without a key it runs offline in mock mode.

The sample corpus contains a fictional company handbook covering security, incident response, leave, engineering standards, onboarding, and expenses.

## How it works

1. **Ingest** splits Markdown and text documents into overlapping chunks, embeds them, and saves a local vector index.
2. **Plan** turns the question into up to three search queries.
3. **Retrieve** combines semantic vector search with BM25 keyword ranking and returns the best passages.
4. **Answer** uses only those passages, cites claims, and abstains if the evidence is insufficient.
5. **Check** reviews a drafted answer for support and can request one revision.

## Run in GitHub Codespaces

The repository currently stores Python modules and sample Markdown files at the repository root, while the code imports them as the `app` package and expects documents under `data/docs`. In a Codespace terminal at the repository root, create the expected runtime layout:

```bash
python -m pip install -r requirements.txt
mkdir -p app data/docs
cp ./*.py app/
cp eval_set.json app/
cp security_policy.md incident_response.md leave_policy.md engineering_standards.md onboarding.md expenses.md data/docs/
```

For Gemini mode, add a GitHub Codespaces secret named `GEMINI_API_KEY` with access to this repository, then restart the Codespace. Do not paste the key into the repository or a chat message.

Build the document index:

```bash
python -m app.cli ingest
```

The default model in this repository may be unavailable to new Gemini API projects. Start the API with a model available to your Gemini project. For example, to use Gemini 3.5 Flash-Lite:

```bash
LLM_MODEL=gemini-3.5-flash-lite python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

In Codespaces, open the forwarded port 8000. Use `/health` to see the active mode and model, and `/docs` to try the API. Expand **POST /ask**, choose **Try it out**, enter a JSON request such as:

```json
{
  "question": "According to the security policy, how soon must suspected phishing emails be reported?"
}
```

Select **Execute** to see the answer, citations, and trace. This repository provides an API, not a chat-style web interface. Codespaces and Gemini free-tier usage have account quotas and rate limits. Do not link billing if you require no-charge usage; requests may stop when free limits are reached.

## Run the evaluation

After the Codespaces layout above is created, run the evaluation with the Gemini key available in the Codespace:

```bash
LLM_MODEL=gemini-3.5-flash-lite python -m app.run_eval --json
```

The report includes retrieval hit rate, answer accuracy, abstention accuracy, p50 and p95 latency, and estimated total cost. The evaluation uses the model/API configuration active in that Codespace.

## Gemini smoke-test result so far

One Gemini `POST /ask` request was tried with the question “What does the security policy say about incident reporting?” The system retrieved passages from `incident_response.md` and `security_policy.md` but abstained. Its trace reported 3.145 seconds latency, two tracked LLM calls, and estimated cost of $0.000238. This is a single smoke test, not an evaluation: accuracy, abstention accuracy, and p50 latency have not been measured. The reported cost is the app's token-price estimate, not a billing statement.

## Mock mode vs. Gemini mode

Without `GEMINI_API_KEY` (or with `USE_MOCK=1`), the app uses a deterministic hashing embedder and a rule-based extractive LLM. Mock mode is useful for checking that the pipeline runs, but it does not measure Gemini answer quality.

If you change embedding models, rebuild the index with `python -m app.cli ingest`. The saved index records its embedder and the app rejects a mismatch.

## Configuration

The app reads configuration from environment variables:

- `GEMINI_API_KEY` or `GOOGLE_API_KEY`: Gemini API key.
- `LLM_MODEL`: generation model (default: `gemini-2.5-flash`).
- `EMBED_MODEL`: embedding model (default: `gemini-embedding-001`).
- `DOCS_DIR`, `INDEX_DIR`, `TRACE_DIR`: documents, index, and trace locations.
- `TOP_K`, `MAX_REVISIONS`: retrieval count and answer-revision limit.

Per-request traces are appended to `traces/traces.jsonl`. The `est_cost_usd` trace field is an estimate based on configurable token prices; check the Gemini pricing page and your AI Studio usage for actual billing and free-tier status.
