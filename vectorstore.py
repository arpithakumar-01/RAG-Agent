"""Hybrid retrieval: dense cosine + BM25, fused with Reciprocal Rank Fusion."""
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import tokenize


@dataclass
class Chunk:
    id: int
    source: str
    section: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    dense: float
    lexical: float
    fused: float


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder_name = embedder_name
        self._tf = [Counter(tokenize(c.text)) for c in chunks]
        self._len = np.array([sum(tf.values()) for tf in self._tf], dtype=np.float32)
        self._avg = float(self._len.mean()) if len(chunks) else 1.0
        df = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def _bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        q = tokenize(query)
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for i, tf in enumerate(self._tf):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self._idf.get(t, 0.0) * f * (k1 + 1) / (f + k1 * (1 - b + b * self._len[i] / self._avg))
            scores[i] = s
        return scores

    def search(self, query: str, qvec: np.ndarray, k: int = 4) -> list[Hit]:
        dense = self.vectors @ qvec
        lex = self._bm25(query)
        rank_d = {int(i): r for r, i in enumerate(np.argsort(-dense))}
        rank_l = {int(i): r for r, i in enumerate(np.argsort(-lex))}
        fused = {i: 1 / (60 + rank_d[i]) + 1 / (60 + rank_l[i]) for i in range(len(self.chunks))}
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.chunks[i], float(dense[i]), float(lex[i]), fused[i]) for i in top]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        meta = {"embedder": self.embedder_name, "chunks": [c.__dict__ for c in self.chunks]}
        (directory / "chunks.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        meta = json.loads((directory / "chunks.json").read_text())
        chunks = [Chunk(**c) for c in meta["chunks"]]
        return cls(chunks, np.load(directory / "vectors.npy"), meta["embedder"])
