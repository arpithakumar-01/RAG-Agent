"""Embedders: Gemini embeddings in production, deterministic hashing embedder offline."""
import re
import zlib

import numpy as np

from . import config

STOP = set(
    "a an the of to in on for and or is are was were be by with as at from that this it its "
    "what how when who which do does can i my we our you your if not no than then so".split()
)


def tokenize(text: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in STOP:
            continue
        if len(t) > 4 and t.endswith("ied"):
            t = t[:-3] + "y"
        elif len(t) > 4 and t.endswith("ed"):
            t = t[:-2]
        elif len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        out.append(t)
    return out


class HashingEmbedder:
    """Signed feature hashing over unigrams+bigrams. No network, no training."""
    name = "hashing-512"
    dim = 512

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = tokenize(text)
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for f in feats:
                h = zlib.crc32(f.encode())
                sign = 1.0 if zlib.crc32((f + "#").encode()) & 1 else -1.0
                mat[i, h % self.dim] += sign
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-9)


class GeminiEmbedder:
    def __init__(self):
        from google import genai
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.name = f"gemini:{config.EMBED_MODEL}"

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        from google.genai import types
        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        vecs = []
        for i in range(0, len(texts), 50):  # batch to stay under request limits
            resp = self.client.models.embed_content(
                model=config.EMBED_MODEL,
                contents=texts[i:i + 50],
                config=types.EmbedContentConfig(task_type=task),
            )
            vecs.extend(e.values for e in resp.embeddings)
        mat = np.array(vecs, dtype=np.float32)
        return mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)


def get_embedder():
    return HashingEmbedder() if config.USE_MOCK else GeminiEmbedder()
