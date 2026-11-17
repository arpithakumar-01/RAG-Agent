"""LLM clients. GeminiLLM for real use; MockLLM is a deterministic offline stand-in
so the whole pipeline (agents, tracing, eval, tests) runs without an API key."""
import json
import re
import time
from dataclasses import dataclass

from . import config
from .embeddings import tokenize

ABSTAIN = "I don't know based on the provided documents."


@dataclass
class LLMResult:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class GeminiLLM:
    def __init__(self):
        from google import genai
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)

    def generate(self, role: str, system: str, prompt: str, json_mode: bool = False,
                 payload: dict = None) -> LLMResult:
        from google.genai import types
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.1,
            response_mime_type="application/json" if json_mode else None,
        )
        last_err = None
        for attempt in range(3):  # retry transient API errors with backoff
            try:
                resp = self.client.models.generate_content(
                    model=config.LLM_MODEL, contents=prompt, config=cfg)
                u = resp.usage_metadata
                return LLMResult(resp.text or "",
                                 getattr(u, "prompt_token_count", 0) or 0,
                                 getattr(u, "candidates_token_count", 0) or 0)
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Gemini call failed after retries: {last_err}")


class MockLLM:
    """Rule-based stand-in. Answers extractively from the retrieved context."""

    @staticmethod
    def _tok(text: str) -> int:
        return int(len(text.split()) * 1.3) + 1

    def generate(self, role: str, system: str, prompt: str, json_mode: bool = False,
                 payload: dict = None) -> LLMResult:
        payload = payload or {}
        if role == "planner":
            q = payload["question"]
            parts = [p.strip(" ?") + "?" for p in re.split(r"\?\s+|\s+and also\s+|\s+and (?=(?:how|what|who|when|where|which)\b)|;\s*", q) if p.strip(" ?")]
            out = json.dumps({"intent": "multi_hop" if len(parts) > 1 else "lookup",
                              "sub_queries": parts[:3] or [q]})
        elif role == "rewriter":
            out = " ".join(tokenize(payload["question"]))
        elif role == "answerer":
            out = self._answer(payload["question"], payload["context"])
        elif role == "critic":
            out = self._critic(payload["answer"], payload["context"])
        else:
            out = ""
        return LLMResult(out, self._tok(prompt + system), self._tok(out))

    def _answer(self, question: str, context: list[tuple[int, str]]) -> str:
        # Company-name words appear in every document, so they carry no evidence of relevance.
        q = set(tokenize(question)) - {"northwind", "lab", "company", "today"}
        quantitative = bool(re.match(r"\s*(how (long|many|much|often|quickly|fast|soon)|when|what is the (minimum|maximum))", question.lower()))
        best_per_passage = []
        for n, text in context:
            cands = []
            for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
                sent = sent.strip(" -*")
                if len(sent.split()) < 3:
                    continue
                overlap = len(q & set(tokenize(sent)))
                bonus = 0.5 if quantitative and re.search(r"\d", sent) else 0.0
                cands.append((overlap + bonus, overlap, sent))
            if cands:
                score, overlap, sent = max(cands, key=lambda x: x[0])
                best_per_passage.append((score, overlap, n, sent))
        best_per_passage.sort(key=lambda x: -x[0])
        if not best_per_passage or best_per_passage[0][1] < 2:
            return ABSTAIN
        top = best_per_passage[0][0]
        multi = question.count("?") > 1 or bool(re.search(r"\band (how|what|who|when|where|which)\b", question.lower()))
        margin = 1.5 if multi else 0.5
        picked = [x for x in best_per_passage[:3] if x[1] >= 2 and x[0] >= top - margin]
        return " ".join(f"{sent.rstrip('.')} [{n}]." for _, _, n, sent in picked)

    def _critic(self, answer: str, context: list[tuple[int, str]]) -> str:
        ctx = dict(context)
        cites = [int(c) for c in re.findall(r"\[(\d+)\]", answer)]
        issues = []
        if not cites:
            issues.append("Answer contains no citations.")
        for c in cites:
            if c not in ctx:
                issues.append(f"Citation [{c}] does not match any context passage.")
        for sent in re.split(r"(?<=\.)\s+", answer):
            m = re.search(r"\[(\d+)\]", sent)
            if m and int(m.group(1)) in ctx:
                words = set(tokenize(re.sub(r"\[\d+\]", "", sent)))
                if words and len(words & set(tokenize(ctx[int(m.group(1))]))) / len(words) < 0.6:
                    issues.append(f"Claim not supported by passage [{m.group(1)}].")
        return json.dumps({"supported": not issues, "issues": issues})


def get_llm():
    return MockLLM() if config.USE_MOCK else GeminiLLM()
