"""Graph-based RAG retriever over the typology knowledge base.

The plain retriever in ``rag.py`` ranks passages by word overlap with the query. This
module adds a graph on top of the same passages:

    chunk ──(mentions, weight)──▶ concept ◀──(activated by case signals)── account evidence
    chunk ◀──(TF-IDF similarity, kNN)──▶ chunk

Case signals (fan-out, a transaction loop, rapid forwarding, ...) activate concept nodes.
Personalized PageRank spreads that activation through the graph, so a passage can rank high
because it shares a concept with the case or sits next to a strongly matching passage, even
when it shares few words with the query. Each result records which concepts pulled it in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .rag import Chunk, KnowledgeBase

# Typology ontology: concept -> phrases that indicate it in a passage.
CONCEPTS: dict[str, list[str]] = {
    "fan_out": ["fan-out", "dispersal", "many counterparties", "distinct recipients", "smurfing", "outbound partners"],
    "fan_in": ["fan-in", "collection", "many senders", "consolidation", "inbound partners"],
    "pass_through": ["pass-through", "pass through", "inflow", "outflow", "layering", "relay", "balance"],
    "rapid_forwarding": ["forwarded", "forwarding", "24 hours", "rapid", "onward transfer", "passed on"],
    "cycle": ["cycle", "circular", "loop", "round-tripping", "origin", "strongly connected"],
    "reciprocity": ["reciprocity", "two-way", "back and forth", "wash trading"],
    "structuring": ["structuring", "threshold", "split", "below"],
    "mule": ["mule", "new account", "spike", "sudden", "withdrawal"],
}

# Case evidence -> concept activation weights in [0, 1].
CHUNK_KNN = 3
KNN_MIN_SIM = 0.15


def _count(text: str, phrase: str) -> int:
    return len(re.findall(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))


def signals_from_case(flags: list[tuple[str, float]] | None, summary: dict | None) -> dict[str, float]:
    """Translate the scorer's flags and the graph summary into concept activations."""
    summary = summary or {}
    act: dict[str, float] = {}

    def bump(concept: str, weight: float) -> None:
        act[concept] = max(act.get(concept, 0.0), min(1.0, weight))

    for name, z in flags or []:
        strength = min(1.0, z / 6.0)
        if name in {"out_partners", "partner_diversity_out", "out_total"}:
            bump("fan_out", strength)
        if name in {"in_partners", "partner_diversity_in", "in_total"}:
            bump("fan_in", strength)
        if name == "in_out_balance":
            bump("pass_through", strength)
        if name == "forward_ratio":
            bump("rapid_forwarding", strength)
        if name == "scc_size":
            bump("cycle", strength)
        if name == "reciprocity":
            bump("reciprocity", strength)
        if name == "active_span_days":
            bump("mule", strength)

    if summary.get("in_cycle"):
        bump("cycle", 1.0)
    if summary.get("reciprocity", 0) > 0.3:
        bump("reciprocity", summary["reciprocity"])
    if summary.get("forward_ratio", 0) > 0.2:
        bump("rapid_forwarding", summary["forward_ratio"])
    if summary.get("out_partners", 0) >= 5:
        bump("fan_out", min(1.0, summary["out_partners"] / 20))
    if summary.get("in_partners", 0) >= 5:
        bump("fan_in", min(1.0, summary["in_partners"] / 20))
    return {k: v for k, v in act.items() if v > 0}


@dataclass
class GraphRetriever:
    kb: KnowledgeBase
    graph: nx.Graph = field(init=False)
    _chunk_nodes: list[str] = field(init=False)
    _concept_weight: dict[tuple[str, str], int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._chunk_nodes = [f"chunk:{c.chunk_id}" for c in self.kb.chunks]
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self._chunk_nodes)
        self.graph.add_nodes_from(f"concept:{name}" for name in CONCEPTS)
        self._link_concepts()
        self._link_similar_chunks()

    def _link_concepts(self) -> None:
        for chunk in self.kb.chunks:
            text = f"{chunk.title} {chunk.text}".lower()
            for concept, phrases in CONCEPTS.items():
                hits = sum(_count(text, p) for p in phrases)
                if hits:
                    weight = min(1.0, hits / 3)
                    self.graph.add_edge(f"chunk:{chunk.chunk_id}", f"concept:{concept}", weight=weight)
                    self._concept_weight[(chunk.chunk_id, concept)] = hits

    def _link_similar_chunks(self) -> None:
        sim = (self.kb._matrix @ self.kb._matrix.T).toarray()  # rows are L2-normalised TF-IDF
        np.fill_diagonal(sim, 0.0)
        for i, chunk in enumerate(self.kb.chunks):
            for j in np.argsort(-sim[i])[:CHUNK_KNN]:
                if sim[i, j] >= KNN_MIN_SIM:
                    self.graph.add_edge(
                        f"chunk:{chunk.chunk_id}", f"chunk:{self.kb.chunks[j].chunk_id}",
                        weight=float(sim[i, j]),
                    )

    def retrieve(self, query: str, signals: dict[str, float] | None = None, k: int = 3) -> list[dict]:
        signals = {c: w for c, w in (signals or {}).items() if c in CONCEPTS and w > 0}
        query_scores = self._query_scores(query)
        if not signals and not query_scores.any():
            return []

        personalization: dict[str, float] = {}
        for concept, w in signals.items():
            personalization[f"concept:{concept}"] = w
        total_q = float(query_scores.sum())
        for node, s in zip(self._chunk_nodes, query_scores):
            if s > 0:
                # Query words seed passages directly; scaled so concepts still dominate when present.
                personalization[node] = 0.5 * float(s) / total_q if total_q else 0.0
        if not personalization or sum(personalization.values()) == 0:
            return []

        ppr = nx.pagerank(self.graph, alpha=0.85, personalization=personalization, weight="weight")
        ranked = sorted(self._chunk_nodes, key=lambda n: -ppr[n])[:k]

        results = []
        for node in ranked:
            if ppr[node] <= 0:
                continue
            chunk = self._chunk_by_node(node)
            results.append({
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "title": chunk.title,
                "text": chunk.text,
                "score": round(float(ppr[node]), 4),
                "tfidf_score": round(float(query_scores[self._index(node)]), 4),
                "via_concepts": self._explain(chunk.chunk_id, signals),
                "retrieval": "graph",
            })
        return results

    # helpers -----------------------------------------------------------------
    def _query_scores(self, query: str) -> np.ndarray:
        if not query.strip():
            return np.zeros(len(self.kb.chunks))
        q = self.kb._vectorizer.transform([query])
        return (self.kb._matrix @ q.T).toarray().ravel()

    def _index(self, node: str) -> int:
        return self._chunk_nodes.index(node)

    def _chunk_by_node(self, node: str) -> Chunk:
        return self.kb.chunks[self._index(node)]

    def _explain(self, chunk_id: str, signals: dict[str, float]) -> list[str]:
        """Concepts that link this passage to the case, strongest first."""
        node = f"chunk:{chunk_id}"
        linked = []
        for concept, w in signals.items():
            if self.graph.has_edge(node, f"concept:{concept}"):
                linked.append((w * self.graph[node][f"concept:{concept}"]["weight"], concept))
        return [c for _, c in sorted(linked, reverse=True)]
