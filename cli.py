"""CLI:  python -m app.cli ingest   |   python -m app.cli ask "your question" """
import argparse
import json

from .ingest import build_index
from .pipeline import RagPipeline


def main():
    p = argparse.ArgumentParser(prog="enterprise-rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="(re)build the vector index from data/docs")
    a = sub.add_parser("ask", help="ask a question")
    a.add_argument("question")
    a.add_argument("--json", action="store_true", help="print the full JSON result")
    args = p.parse_args()

    if args.cmd == "ingest":
        store = build_index()
        print(f"Indexed {len(store.chunks)} chunks with embedder '{store.embedder_name}'.")
        return
    res = RagPipeline().ask(args.question)
    if args.json:
        print(json.dumps(res, indent=2))
        return
    t = res["trace"]
    print(f"\nQ: {res['question']}\nA: {res['answer']}\n")
    for c in res["citations"]:
        print(f"  [{c['n']}] {c['source']} > {c['section']}")
    print(f"\nverified={res['verified']}  latency={t['latency_s']}s  llm_calls={t['llm_calls']}  "
          f"tokens={t['tokens_in']}/{t['tokens_out']}  est_cost=${t['est_cost_usd']}")


if __name__ == "__main__":
    main()
