"""Lightweight per-request tracing: spans, token counts, cost, tokens/sec."""
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class Span:
    name: str
    start: float
    end: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def seconds(self) -> float:
        return max(self.end - self.start, 0.0)


class Trace:
    def __init__(self, question: str):
        self.id = uuid.uuid4().hex[:12]
        self.question = question
        self.spans: list[Span] = []
        self._t0 = time.perf_counter()

    @contextmanager
    def span(self, name: str, **meta):
        s = Span(name=name, start=time.perf_counter(), meta=dict(meta))
        try:
            yield s
        finally:
            s.end = time.perf_counter()
            self.spans.append(s)

    def summary(self) -> dict:
        llm = [s for s in self.spans if s.tokens_in or s.tokens_out]
        t_in = sum(s.tokens_in for s in llm)
        t_out = sum(s.tokens_out for s in llm)
        gen_time = sum(s.seconds for s in llm)
        cost = t_in / 1e6 * config.PRICE_IN_PER_M + t_out / 1e6 * config.PRICE_OUT_PER_M
        return {
            "trace_id": self.id,
            "latency_s": round(time.perf_counter() - self._t0, 4),
            "llm_calls": len(llm),
            "tokens_in": t_in,
            "tokens_out": t_out,
            "tokens_per_sec": round(t_out / gen_time, 2) if gen_time > 0 else 0.0,
            "est_cost_usd": round(cost, 6),
            "spans": [
                {"name": s.name, "seconds": round(s.seconds, 4), "tokens_in": s.tokens_in,
                 "tokens_out": s.tokens_out, **s.meta}
                for s in self.spans
            ],
        }

    def flush(self) -> None:
        try:
            Path(config.TRACE_DIR).mkdir(parents=True, exist_ok=True)
            with open(Path(config.TRACE_DIR) / "traces.jsonl", "a") as f:
                f.write(json.dumps({"question": self.question, **self.summary()}) + "\n")
        except OSError:
            pass  # tracing must never break a request
