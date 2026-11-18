"""Retrieval over the AML typology knowledge base (docs/kb/*.md).

Each markdown file is split on "## " headings into chunks. Retrieval uses
TF-IDF cosine similarity, so it runs offline with no embedding model download.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[3]
KB_DIR = ROOT / "docs" / "kb"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    title: str
    text: str


def load_chunks(kb_dir: Path = KB_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        doc_title = path.stem.replace("_", " ")
        sections = path.read_text(encoding="utf-8").split("\n## ")
        for i, section in enumerate(sections):
            body = section.strip()
            if not body or body.startswith("# "):
                # Document header block: skip if it has no body text of its own.
                head, _, rest = body.partition("\n")
                if not rest.strip():
                    continue
                title, text = head.lstrip("# ").strip(), rest.strip()
            else:
                title, _, text = body.partition("\n")
                title, text = title.strip(), text.strip()
            if text:
                chunks.append(Chunk(f"{path.stem}#{i}", path.name, title or doc_title, text))
    return chunks


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("knowledge base is empty")
        self.chunks = chunks
        self._vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self._matrix = self._vectorizer.fit_transform(
            [f"{c.title}. {c.text}" for c in chunks]
        )

    @classmethod
    def from_dir(cls, kb_dir: Path = KB_DIR) -> "KnowledgeBase":
        return cls(load_chunks(kb_dir))

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not query.strip():
            return []
        q = self._vectorizer.transform([query])
        scores = (self._matrix @ q.T).toarray().ravel()
        order = scores.argsort()[::-1][:k]
        results = []
        for idx in order:
            if scores[idx] <= 0:
                continue
            c = self.chunks[idx]
            results.append(
                {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "title": c.title,
                    "text": c.text,
                    "score": round(float(scores[idx]), 4),
                }
            )
        return results
