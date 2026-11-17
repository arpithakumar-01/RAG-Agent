"""Central configuration. Everything is overridable via environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(os.getenv("DOCS_DIR", ROOT / "data" / "docs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "index"))
TRACE_DIR = Path(os.getenv("TRACE_DIR", ROOT / "traces"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

# With no API key (or USE_MOCK=1) the system runs fully offline using a
# deterministic mock LLM + hashing embedder. Used for tests/CI and demos.
USE_MOCK = os.getenv("USE_MOCK", "").lower() in ("1", "true", "yes") or not GEMINI_API_KEY

CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "110"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "1"))   # self-reflection loops
MIN_LEXICAL_SCORE = float(os.getenv("MIN_LEXICAL_SCORE", "1.5"))  # below -> rewrite query / abstain

# USD per 1M tokens. These are configurable ESTIMATES: check current Gemini pricing.
PRICE_IN_PER_M = float(os.getenv("PRICE_IN_PER_M", "0.30"))
PRICE_OUT_PER_M = float(os.getenv("PRICE_OUT_PER_M", "2.50"))
