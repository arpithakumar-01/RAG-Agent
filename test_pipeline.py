import json
from pathlib import Path

import pytest

from app import config
from app.ingest import build_index, load_chunks, window
from app.pipeline import RagPipeline


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    config.USE_MOCK = True
    config.TRACE_DIR = tmp_path_factory.mktemp("traces")
    store = build_index(config.DOCS_DIR, tmp_path_factory.mktemp("index"))
    return RagPipeline(store=store)


def test_chunking_respects_size_and_overlap():
    words = [str(i) for i in range(300)]
    chunks = window(words, 100, 20)
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[1][0] == "80"  # 20-word overlap


def test_all_docs_ingested():
    sources = {c.source for c in load_chunks(config.DOCS_DIR)}
    assert len(sources) == 6


def test_grounded_answer_with_citation(pipeline):
    r = pipeline.ask("How long must a password be?")
    assert "14" in r["answer"] and r["verified"] and r["citations"]
    assert "security_policy.md" in {c["source"] for c in r["citations"]}


def test_multi_hop_question_hits_both_documents(pipeline):
    r = pipeline.ask("How often is the SEV1 status page updated? How many days of sick leave are there?")
    assert {"incident_response.md", "leave_policy.md"} <= set(r["retrieved_sources"])


def test_out_of_scope_question_abstains(pipeline):
    r = pipeline.ask("What is the CEO's favourite colour?")
    assert r["abstained"] and not r["citations"]


def test_trace_has_llm_native_metrics(pipeline):
    t = pipeline.ask("What is the minimum unit test coverage?")["trace"]
    assert t["llm_calls"] >= 3 and t["tokens_in"] > 0 and t["est_cost_usd"] >= 0
    assert {"planner", "retriever", "answerer", "critic"} <= {s["name"] for s in t["spans"]}


def test_eval_set_passes(pipeline):
    from eval.run_eval import run
    cases = json.loads((Path(config.ROOT) / "eval" / "eval_set.json").read_text())
    rep = run(pipeline, cases)
    assert rep["answer_accuracy"] >= 0.85 and rep["abstention_accuracy"] == 1.0, rep["failures"]
