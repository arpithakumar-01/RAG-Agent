"""Orchestrator: wires the agents together and returns an answer plus a full trace."""
import re
from pathlib import Path

from . import config
from .agents import AnswerAgent, CriticAgent, PlannerAgent, RetrieverAgent
from .embeddings import get_embedder
from .ingest import build_index
from .llm import ABSTAIN, get_llm
from .tracing import Trace
from .vectorstore import VectorStore


class RagPipeline:
    def __init__(self, store: VectorStore = None, embedder=None, llm=None):
        self.embedder = embedder or get_embedder()
        self.llm = llm or get_llm()
        if store is None:
            idx = Path(config.INDEX_DIR)
            store = VectorStore.load(idx) if (idx / "chunks.json").exists() else build_index()
        if store.embedder_name != self.embedder.name:
            raise RuntimeError(
                f"Index was built with '{store.embedder_name}' but the active embedder is "
                f"'{self.embedder.name}'. Re-run ingestion (python -m app.cli ingest).")
        self.store = store
        self.planner = PlannerAgent(self.llm)
        self.retriever = RetrieverAgent(store, self.embedder, self.llm)
        self.answerer = AnswerAgent(self.llm)
        self.critic = CriticAgent(self.llm)

    def ask(self, question: str) -> dict:
        trace = Trace(question)
        plan = self.planner.run(question, trace)
        hits = self.retriever.run(plan, trace)

        answer, feedback, verified = "", None, False
        for _ in range(config.MAX_REVISIONS + 1):          # self-reflection loop
            answer = self.answerer.run(question, hits, trace, feedback)
            if answer.startswith(ABSTAIN):
                verified = True                             # abstaining makes no claims to verify
                break
            verdict = self.critic.run(answer, hits, trace)
            verified = bool(verdict.get("supported"))
            if verified:
                break
            feedback = [str(i) for i in verdict.get("issues", [])] or ["Answer not fully supported."]

        abstained = answer.startswith(ABSTAIN)
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(hits)})
        citations = [{"n": n, "source": hits[n - 1].chunk.source, "section": hits[n - 1].chunk.section,
                      "snippet": hits[n - 1].chunk.text[:200]} for n in cited]
        trace.flush()
        return {"question": question, "answer": answer, "abstained": abstained, "verified": verified,
                "citations": citations, "sub_queries": plan.sub_queries,
                "retrieved_sources": [h.chunk.source for h in hits], "trace": trace.summary()}
