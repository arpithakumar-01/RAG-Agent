"""Evaluation harness. Reports retrieval hit-rate, answer correctness, abstention accuracy,
groundedness (verified rate), latency, token use and estimated cost.
Usage: python -m eval.run_eval [--json]"""
import argparse
import json
import statistics
from pathlib import Path

from app.pipeline import RagPipeline


def run(pipeline: RagPipeline, cases: list[dict]) -> dict:
    rows = []
    for c in cases:
        res = pipeline.ask(c["q"])
        t = res["trace"]
        row = {"q": c["q"], "latency": t["latency_s"], "cost": t["est_cost_usd"],
               "tokens": t["tokens_in"] + t["tokens_out"], "verified": res["verified"]}
        if c.get("expect_abstain"):
            row["kind"], row["pass"] = "abstain", res["abstained"]
        else:
            srcs = {c["source"], c.get("also_source")} - {None}
            row["kind"] = "answer"
            row["retrieval_hit"] = srcs.issubset(set(res["retrieved_sources"]))
            low = res["answer"].lower()
            row["answer_ok"] = (not res["abstained"]) and all(s.lower() in low for s in c["answer_contains"])
            row["pass"] = row["retrieval_hit"] and row["answer_ok"]
        rows.append(row)
    ans = [r for r in rows if r["kind"] == "answer"]
    abst = [r for r in rows if r["kind"] == "abstain"]
    lat = sorted(r["latency"] for r in rows)
    return {
        "n": len(rows),
        "retrieval_hit_rate": round(sum(r["retrieval_hit"] for r in ans) / max(len(ans), 1), 3),
        "answer_accuracy": round(sum(r["answer_ok"] for r in ans) / max(len(ans), 1), 3),
        "abstention_accuracy": round(sum(r["pass"] for r in abst) / max(len(abst), 1), 3),
        "verified_rate": round(sum(r["verified"] for r in rows) / len(rows), 3),
        "latency_p50_s": round(statistics.median(lat), 4),
        "latency_p95_s": round(lat[min(int(len(lat) * 0.95), len(lat) - 1)], 4),
        "avg_tokens": round(statistics.mean(r["tokens"] for r in rows), 1),
        "total_est_cost_usd": round(sum(r["cost"] for r in rows), 6),
        "failures": [r["q"] for r in rows if not r["pass"]],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cases = json.loads((Path(__file__).parent / "eval_set.json").read_text())
    report = run(RagPipeline(), cases)
    print(json.dumps(report, indent=2))
