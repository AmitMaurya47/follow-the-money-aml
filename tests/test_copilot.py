"""Tests for the graph + ML + RAG + multi-agent copilot. Offline by default."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from aml_copilot.copilot.agents import build_app, investigate
from aml_copilot.copilot.data import account_labels, load_transactions
from aml_copilot.copilot.graph_features import account_features, build_graph
from aml_copilot.copilot.model import FEATURES, fit_model
from aml_copilot.copilot.rag import KnowledgeBase, load_chunks

HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering"
)


def _write_csv(tmp_path, rows):
    path = tmp_path / "txns.csv"
    lines = [HEADER] + [
        f"{ts},{fb},{fa},{tb},{ta},{amt},US Dollar,{amt},US Dollar,ACH,{lab}"
        for ts, fb, fa, tb, ta, amt, lab in rows
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture()
def tiny_df(tmp_path):
    rows = [
        # Cycle A -> B -> C -> A (laundering)
        ("2022/09/01 00:00", 1, "A", 1, "B", 100, 1),
        ("2022/09/01 01:00", 1, "B", 1, "C", 100, 1),
        ("2022/09/01 02:00", 1, "C", 1, "A", 100, 1),
        # Pass-through P: receives 500 from X, forwards 480 to Y within 2h
        ("2022/09/02 00:00", 1, "X", 1, "P", 500, 0),
        ("2022/09/02 02:00", 1, "P", 1, "Y", 480, 0),
        # Fan-out F: sends small amounts to six distinct accounts
        *[("2022/09/03 0%d:00" % i, 1, "F", 1, f"D{i}", 50, 0) for i in range(6)],
        # Ordinary background traffic
        ("2022/09/04 00:00", 1, "Q", 1, "R", 20, 0),
    ]
    return load_transactions(_write_csv(tmp_path, rows))


def test_load_transactions_normalizes_columns(tiny_df):
    assert {"src", "dst", "amount_paid", "is_laundering", "timestamp"} <= set(tiny_df.columns)
    assert tiny_df["timestamp"].is_monotonic_increasing
    assert tiny_df.loc[0, "src"] == "1:A"  # earliest transfer is A -> B


def test_load_transactions_rejects_wrong_layout(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError):
        load_transactions(bad)


def test_account_labels_cover_both_sides_of_laundering(tiny_df):
    labels = account_labels(tiny_df)
    illicit = set(labels[labels == 1].index)
    assert illicit == {"1:A", "1:B", "1:C"}


def test_graph_features_detect_cycle_passthrough_and_fanout(tiny_df):
    feats = account_features(tiny_df, build_graph(tiny_df))
    assert feats.loc["1:A", "scc_size"] == 3
    assert feats.loc["1:Q", "scc_size"] == 0
    assert feats.loc["1:P", "forward_ratio"] == 1.0
    assert feats.loc["1:P", "in_out_balance"] > 0.9
    assert feats.loc["1:F", "out_partners"] == 6
    assert feats.loc["1:F", "partner_diversity_out"] == 1.0


def test_knowledge_base_retrieves_matching_typology(tmp_path):
    (tmp_path / "fan.md").write_text(
        "# Fan\n\nintro\n\n## Fan-out dispersal\nOne account sends funds to many counterparties.\n"
    )
    (tmp_path / "loop.md").write_text(
        "# Loop\n\n## Circular flows\nMoney returns to its origin through a loop of accounts.\n"
    )
    kb = KnowledgeBase(load_chunks(tmp_path))
    hits = kb.search("many counterparties fan-out dispersal", k=2)
    assert hits and hits[0]["source"] == "fan.md"
    assert kb.search("") == []


def test_shipped_knowledge_base_loads():
    kb = KnowledgeBase.from_dir()
    hits = kb.search("circular loop round-tripping cycle", k=1)
    assert hits and "circular" in hits[0]["source"]


def test_fit_model_produces_scores_and_metrics(tiny_df):
    # Larger synthetic population so the train/test split is meaningful.
    import numpy as np

    rng = np.random.default_rng(0)
    n = 300
    feats = pd.DataFrame(
        {f: rng.normal(size=n) for f in FEATURES},
        index=[f"acc{i}" for i in range(n)],
    )
    labels = pd.Series((feats["pagerank"] + feats["scc_size"] > 0.8).astype(int).values, index=feats.index)
    model, metrics = fit_model(feats, labels)
    assert 0.0 <= model.predict_proba(feats.iloc[[0]]) <= 1.0
    assert metrics["roc_auc"] > 0.7
    assert {"pr_auc", "precision_at_top_1pct", "n_test"} <= set(metrics)


@pytest.fixture()
def synthetic_bundle(tiny_df):
    import numpy as np

    graph = build_graph(tiny_df)
    feats = account_features(tiny_df, graph)
    labels = account_labels(tiny_df)
    rng = np.random.default_rng(1)
    pad = pd.DataFrame(
        {f: rng.normal(size=200) for f in FEATURES},
        index=[f"pad{i}" for i in range(200)],
    )
    all_feats = pd.concat([feats, pad])
    all_labels = pd.concat([labels, pd.Series(0, index=pad.index)])
    model, _ = fit_model(all_feats, all_labels)
    return {
        "model": model,
        "features": all_feats,
        "graph": graph,
        "kb": KnowledgeBase.from_dir(),
    }


class _FakeLLM:
    def __init__(self, text=None, error=None):
        self.text, self.error, self.calls = text, error, 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)])


class _StubModel:
    """Fixed-score model so each routing branch can be forced deterministically."""

    def __init__(self, score):
        self.score = score

    def predict_proba(self, row):
        return self.score

    def elevated_features(self, row, z_min=3.0):
        return [("out_partners", 4.2)] if self.score >= 0.5 else []


def _with_score(bundle, score):
    return {**bundle, "model": _StubModel(score)}


def test_low_risk_account_takes_short_route(synthetic_bundle):
    out = investigate(_with_score(synthetic_bundle, 0.05), "1:A")
    assert out["disposition"] == "CLEAR"
    assert out["writer_mode"] == "none"
    assert "graph_summary" not in out
    assert not any(t.startswith("retriever") for t in out["trace"])


def test_high_risk_account_runs_full_pipeline_with_template(synthetic_bundle):
    out = investigate(_with_score(synthetic_bundle, 0.9), "1:A", llm=None)
    assert out["disposition"] == "ESCALATE"
    assert out["writer_mode"] == "template"
    assert out["graph_summary"]["in_cycle"] is True  # 1:A sits in the A-B-C loop
    assert out["evidence"], "retriever should attach knowledge-base passages"
    assert "[KB1]" in out["narrative"]
    assert "does not establish that a crime occurred" in out["narrative"]


def test_mid_risk_account_clears_below_decision_threshold(synthetic_bundle):
    out = investigate(_with_score(synthetic_bundle, 0.3), "1:A")
    assert out["disposition"] == "CLEAR"
    assert out["writer_mode"] == "template"


def test_llm_success_path_uses_claude_text(synthetic_bundle):
    fake = _FakeLLM(text="Draft narrative citing [KB1].")
    out = investigate(_with_score(synthetic_bundle, 0.9), "1:A", llm=fake)
    assert out["narrative"] == "Draft narrative citing [KB1]."
    assert out["writer_mode"] == "llm"
    assert fake.calls == 1


def test_llm_failure_falls_back_to_template(synthetic_bundle):
    fake = _FakeLLM(error=RuntimeError("network down"))
    out = investigate(_with_score(synthetic_bundle, 0.9), "1:A", llm=fake)
    assert out["writer_mode"] == "template"
    assert any("llm failed" in t for t in out["trace"])
    assert out["disposition"] == "ESCALATE"


def test_unknown_account_raises(synthetic_bundle):
    with pytest.raises(KeyError):
        investigate(synthetic_bundle, "does-not-exist")


def test_stream_emits_nodes_in_order_then_final_state(synthetic_bundle):
    from aml_copilot.copilot.agents import DONE_KEY, stream_investigation

    events = list(stream_investigation(_with_score(synthetic_bundle, 0.9), "1:A"))
    nodes = [n for n, _ in events if n != DONE_KEY]
    assert nodes == ["scorer", "graph_investigator", "retriever", "writer", "decide"]
    final = events[-1][1]
    assert events[-1][0] == DONE_KEY
    assert final["disposition"] == "ESCALATE"
    assert final["account_id"] == "1:A"
    assert len(final["trace"]) == len(nodes)


def test_stream_short_route_skips_investigation_nodes(synthetic_bundle):
    from aml_copilot.copilot.agents import DONE_KEY, stream_investigation

    nodes = [n for n, _ in stream_investigation(_with_score(synthetic_bundle, 0.05), "1:A") if n != DONE_KEY]
    assert nodes == ["scorer", "decide"]


def test_viz_figures_build(synthetic_bundle):
    from aml_copilot.copilot.viz import (
        confusion_figure,
        ego_network_figure,
        risk_gauge_figure,
        score_distribution_figure,
        workflow_figure,
    )

    scores = pd.Series({a: 0.1 + 0.8 * (i % 5) / 5 for i, a in enumerate(synthetic_bundle["features"].index)})
    labels = pd.Series(0, index=scores.index)
    labels["1:A"] = 1
    graph = synthetic_bundle["graph"]

    wf = workflow_figure({"scorer": "done", "graph_investigator": "running"}, {"scorer": "score 0.9"})
    assert len(wf.data) == 1 and len(wf.layout.annotations) >= 5
    ego = ego_network_figure(graph, scores, labels, "1:A", hops=2, max_nodes=10)
    node_trace = ego.data[-1]
    assert "1:A" in [str(t) for t in node_trace.hovertext[0:1]] or any("1:A" in str(h) for h in node_trace.hovertext)
    assert len(node_trace.x) <= 10
    assert score_distribution_figure(scores, labels).data
    assert risk_gauge_figure(0.7).data
    assert confusion_figure(1, 2, 3, 4).data[0].z == ((3, 2), (4, 1)) or list(map(list, confusion_figure(1, 2, 3, 4).data[0].z)) == [[3, 2], [4, 1]]
