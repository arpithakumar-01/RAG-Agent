"""Ingestion pipeline: load docs -> heading-aware chunking -> embed -> persist index."""
import re
from pathlib import Path

from . import config
from .embeddings import get_embedder
from .vectorstore import Chunk, VectorStore


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown on headings; returns (section_title, body)."""
    sections, title, buf = [], "Introduction", []
    for line in text.splitlines():
        m = re.match(r"^#{1,3}\s+(.*)", line)
        if m:
            if "".join(buf).strip():
                sections.append((title, "\n".join(buf).strip()))
            title, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((title, "\n".join(buf).strip()))
    return sections


def window(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = max(size - overlap, 1)
    return [words[i:i + size] for i in range(0, len(words) - overlap, step)]


def load_chunks(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    paths = sorted(Path(docs_dir).glob("*.md")) + sorted(Path(docs_dir).glob("*.txt"))
    for path in paths:
        for title, body in split_sections(path.read_text(encoding="utf-8")):
            for w in window(body.split(), config.CHUNK_WORDS, config.CHUNK_OVERLAP):
                chunks.append(Chunk(len(chunks), path.name, title, " ".join(w)))
    return chunks


def build_index(docs_dir: Path = None, index_dir: Path = None) -> VectorStore:
    docs_dir, index_dir = Path(docs_dir or config.DOCS_DIR), Path(index_dir or config.INDEX_DIR)
    chunks = load_chunks(docs_dir)
    if not chunks:
        raise ValueError(f"No .md/.txt documents found in {docs_dir}")
    embedder = get_embedder()
    # Prefix chunks with source+section so headings contribute to retrieval.
    vecs = embedder.embed([f"{c.source} | {c.section}\n{c.text}" for c in chunks], kind="document")
    store = VectorStore(chunks, vecs, embedder.name)
    store.save(index_dir)
    return store
