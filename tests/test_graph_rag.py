"""Tests for the graph-based RAG retriever. Offline; one test uses the built artifacts if present."""
from __future__ import annotations

import pytest

from aml_copilot.copilot.rag import KnowledgeBase
from aml_copilot.copilot.rag_graph import CONCEPTS, GraphRetriever, signals_from_case


@pytest.fixture(scope="module")
def retriever():
    return GraphRetriever(KnowledgeBase.from_dir())


def test_signals_from_flags_and_summary():
    sig = signals_from_case(
        [("out_partners", 6.0), ("scc_size", 3.0)],
        {"in_cycle": True, "reciprocity": 0.0, "forward_ratio": 0.0},
    )
    assert sig["fan_out"] == 1.0
    assert sig["cycle"] == 1.0
    assert set(sig) <= set(CONCEPTS)


def test_no_evidence_gives_no_signals():
    assert signals_from_case([], {}) == {}


def test_graph_connects_passages_to_concepts(retriever):
    g = retriever.graph
    assert g.has_edge("chunk:typologies_fan_in_out.md#1", "concept:fan_out") or any(
        g.has_edge(n, "concept:fan_out") for n in g.neighbors("concept:fan_out")
    )
    assert any(g.has_edge(n, "concept:cycle") for n in g.nodes if str(n).startswith("chunk:"))


def test_fan_out_signal_ranks_fan_out_passage_first(retriever):
    hits = retriever.retrieve("", {"fan_out": 1.0}, k=3)
    assert hits and hits[0]["source"] == "typologies_fan_in_out.md"
    assert "fan_out" in hits[0]["via_concepts"]
    assert all(h["retrieval"] == "graph" for h in hits)


def test_cycle_signal_ranks_circular_passage_first(retriever):
    hits = retriever.retrieve("", {"cycle": 1.0}, k=2)
    assert hits and hits[0]["source"] == "typologies_circular_flows.md"


def test_query_only_retrieval_still_works(retriever):
    hits = retriever.retrieve("money mule new account spike", {}, k=2)
    assert hits and hits[0]["source"] == "money_mules_and_sar_guidance.md"


def test_empty_request_returns_nothing(retriever):
    assert retriever.retrieve("", {}, k=3) == []


def test_propagation_reaches_passages_without_direct_concept(retriever):
    # Concept activation alone should pull in passages connected through similar-chunk edges.
    hits = retriever.retrieve("", {"pass_through": 1.0, "rapid_forwarding": 1.0}, k=6)
    sources = {h["source"] for h in hits}
    assert "typologies_layering.md" in sources


def test_real_bundle_investigation_uses_graph_retrieval():
    from aml_copilot.copilot import ARTIFACTS, RAW_CSV, load_bundle
    from aml_copilot.copilot.agents import investigate

    if not (ARTIFACTS / "model.joblib").exists() or not RAW_CSV.exists():
        pytest.skip("copilot artifacts not built")
    bundle = load_bundle()
    account = bundle.features["out_partners"].idxmax()
    out = investigate(bundle.as_agent_bundle(), account)
    assert out["retrieval_mode"] == "graph"
    assert out["evidence"]
