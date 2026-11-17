"""Four cooperating agents: Planner -> Retriever (ReAct-style) -> Answerer <-> Critic (self-reflection)."""
import json
import re
from dataclasses import dataclass

from . import config
from .embeddings import tokenize
from .llm import ABSTAIN
from .tracing import Trace
from .vectorstore import Hit


def parse_json(text: str, default: dict) -> dict:
    """LLMs sometimes wrap JSON in prose or code fences; extract the first object."""
    try:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else default
    except (json.JSONDecodeError, TypeError):
        return default


@dataclass
class Plan:
    intent: str
    sub_queries: list[str]


class PlannerAgent:
    SYSTEM = ("You plan retrieval for a company-documents assistant. Break the user's question into at most "
              "3 standalone search queries only if it asks about several distinct things; otherwise return one. "
              'Reply ONLY with JSON: {"intent": "lookup"|"multi_hop", "sub_queries": [str, ...]}')

    def __init__(self, llm):
        self.llm = llm

    def run(self, question: str, trace: Trace) -> Plan:
        with trace.span("planner") as sp:
            r = self.llm.generate("planner", self.SYSTEM, f"Question: {question}",
                                  json_mode=True, payload={"question": question})
            sp.tokens_in, sp.tokens_out = r.tokens_in, r.tokens_out
            data = parse_json(r.text, {})
            subs = [s for s in data.get("sub_queries", []) if isinstance(s, str) and s.strip()][:3]
            plan = Plan(data.get("intent", "lookup"), subs or [question])
            sp.meta["sub_queries"] = plan.sub_queries
        return plan


class RetrieverAgent:
    """ReAct-style loop: Thought (is retrieval confident?) -> Action (rewrite query) -> Observation."""
    REWRITE_SYSTEM = ("Rewrite the search query to be a short keyword-style query using terminology likely "
                      "found in internal policy documents. Reply with the query only.")

    def __init__(self, store, embedder, llm):
        self.store, self.embedder, self.llm = store, embedder, llm

    def _search(self, q: str, k: int) -> list[Hit]:
        return self.store.search(q, self.embedder.embed([q], kind="query")[0], k=k)

    def run(self, plan: Plan, trace: Trace, k: int = None) -> list[Hit]:
        k = k or config.TOP_K
        merged: dict[int, Hit] = {}
        for q in plan.sub_queries:
            steps = []
            with trace.span("retriever", query=q) as sp:
                hits = self._search(q, k)
                best = max((h.lexical for h in hits), default=0.0)
                steps.append(f"Observation: best lexical score {best:.2f}")
                if best < config.MIN_LEXICAL_SCORE:
                    steps.append("Thought: weak match, try a reformulated query")
                    r = self.llm.generate("rewriter", self.REWRITE_SYSTEM, q, payload={"question": q})
                    sp.tokens_in, sp.tokens_out = r.tokens_in, r.tokens_out
                    q2 = r.text.strip() or q
                    steps.append(f"Action: rewrite -> '{q2}'")
                    hits2 = self._search(q2, k)
                    best2 = max((h.lexical for h in hits2), default=0.0)
                    steps.append(f"Observation: best lexical score {best2:.2f}")
                    if best2 > best:
                        hits = hits2
                sp.meta["react_steps"] = steps
            for h in hits:
                cur = merged.get(h.chunk.id)
                if cur is None or h.fused > cur.fused:
                    merged[h.chunk.id] = h
        cap = k + len(plan.sub_queries) - 1
        return sorted(merged.values(), key=lambda h: -h.fused)[:cap]


class AnswerAgent:
    SYSTEM = ("You answer questions using ONLY the numbered context passages. Cite every claim with its "
              f"passage number like [1]. If the context does not contain the answer, reply exactly: {ABSTAIN} "
              "Be concise. Never use outside knowledge.")

    def __init__(self, llm):
        self.llm = llm

    def run(self, question: str, hits: list[Hit], trace: Trace, feedback: list[str] = None) -> str:
        context = [(i + 1, h.chunk.text) for i, h in enumerate(hits)]
        blocks = "\n\n".join(f"[{i + 1}] ({h.chunk.source} > {h.chunk.section})\n{h.chunk.text}"
                             for i, h in enumerate(hits))
        prompt = f"Context:\n{blocks}\n\nQuestion: {question}"
        if feedback:
            prompt += "\n\nA reviewer found problems with your previous answer. Fix them:\n- " + "\n- ".join(feedback)
        with trace.span("answerer", revision=bool(feedback)) as sp:
            r = self.llm.generate("answerer", self.SYSTEM, prompt,
                                  payload={"question": question, "context": context})
            sp.tokens_in, sp.tokens_out = r.tokens_in, r.tokens_out
        return r.text.strip()


class CriticAgent:
    SYSTEM = ("You are a strict fact-checker. Given context passages and an answer, decide whether EVERY claim "
              'in the answer is directly supported by the cited passage. Reply ONLY with JSON: '
              '{"supported": true|false, "issues": [str, ...]}')

    def __init__(self, llm):
        self.llm = llm

    def run(self, answer: str, hits: list[Hit], trace: Trace) -> dict:
        context = [(i + 1, h.chunk.text) for i, h in enumerate(hits)]
        blocks = "\n\n".join(f"[{n}] {t}" for n, t in context)
        with trace.span("critic") as sp:
            r = self.llm.generate("critic", self.SYSTEM, f"Context:\n{blocks}\n\nAnswer:\n{answer}",
                                  json_mode=True, payload={"answer": answer, "context": context})
            sp.tokens_in, sp.tokens_out = r.tokens_in, r.tokens_out
            verdict = parse_json(r.text, {"supported": False, "issues": ["Critic returned invalid JSON."]})
            sp.meta["supported"] = bool(verdict.get("supported"))
        return verdict
