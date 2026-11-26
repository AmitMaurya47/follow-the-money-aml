"""Run logging, crash recovery and the responsible-AI guard for the agent workflow."""
import json

import pytest

from aml_copilot.copilot import agents
from aml_copilot.copilot.agents import DONE_KEY, resume_investigation, stream_investigation
from aml_copilot.copilot.runs import RunStore
from aml_copilot.copilot.rag import KnowledgeBase

# Reuse the synthetic bundle and stubs from the copilot tests.
from test_copilot import _FakeLLM, _with_score, synthetic_bundle, tiny_df  # noqa: F401


def _nodes_started(store, run_id):
    return [e["node"] for e in store.events(run_id) if e["event"] == "node_start"]


def test_completed_run_is_logged_and_checkpointed(synthetic_bundle, tmp_path):
    store = RunStore(tmp_path)
    events = list(stream_investigation(_with_score(synthetic_bundle, 0.9), "1:A", store=store))
    final = events[-1][1]
    run_id = final["run_id"]

    record = store.load(run_id)
    assert record.status == "completed"
    assert record.account_id == "1:A"
    assert record.last_node == "decide"
    assert _nodes_started(store, run_id) == ["scorer", "graph_investigator", "retriever", "writer", "decide"]
    done = [e for e in store.events(run_id) if e["event"] == "run_completed"][0]
    assert done["advisory_only"] is True and done["human_review_required"] is True
    assert final["review_note"].startswith("Advisory output")


def test_log_holds_metadata_not_narrative(synthetic_bundle, tmp_path):
    store = RunStore(tmp_path)
    llm = _FakeLLM(text="The account shows a fan-out pattern [KB1].")
    final = list(stream_investigation(_with_score(synthetic_bundle, 0.9), "1:A", llm=llm, store=store))[-1][1]
    raw = (tmp_path / f"{final['run_id']}.jsonl").read_text()
    assert "fan-out pattern" not in raw  # narrative text stays out of the log
    assert "api_key" not in raw.lower()  # no credentials in the log


def test_crash_in_retriever_then_resume_skips_finished_agents(synthetic_bundle, tmp_path, monkeypatch):
    store = RunStore(tmp_path)
    real_search = KnowledgeBase.search
    calls = {"n": 0}

    def flaky_search(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("index unavailable")
        return real_search(self, *args, **kwargs)

    monkeypatch.setattr(KnowledgeBase, "search", flaky_search)
    bundle = _with_score(synthetic_bundle, 0.9)
    run_id = None
    with pytest.raises(RuntimeError, match="index unavailable"):
        for node, update in stream_investigation(bundle, "1:A", store=store):
            pass
    run_id = store.list_runs()[0]

    record = store.load(run_id)
    assert record.status == "failed"
    assert record.failed_node == "retriever"
    assert record.last_node == "graph_investigator"
    assert "index unavailable" in record.error

    events = list(resume_investigation(bundle, run_id, store=store))
    final = events[-1][1]
    assert final["disposition"] == "ESCALATE"
    assert final["run_id"] == run_id
    assert store.load(run_id).status == "completed"

    started = _nodes_started(store, run_id)
    # scorer and graph_investigator ran once; retriever ran twice (crash, then retry).
    assert started.count("scorer") == 1
    assert started.count("graph_investigator") == 1
    assert started.count("retriever") == 2
    assert [e for e in store.events(run_id) if e["event"] == "run_resumed"][0]["after_node"] == "graph_investigator"


def test_crash_in_first_agent_restarts_same_run(synthetic_bundle, tmp_path, monkeypatch):
    store = RunStore(tmp_path)
    bundle = _with_score(synthetic_bundle, 0.9)
    monkeypatch.setattr(bundle["model"], "predict_proba", lambda row: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        list(stream_investigation(bundle, "1:A", store=store))
    run_id = store.list_runs()[0]
    assert store.load(run_id).checkpoint is None

    monkeypatch.undo()
    final = list(resume_investigation(_with_score(synthetic_bundle, 0.9), run_id, store=store))[-1][1]
    assert final["disposition"] == "ESCALATE"
    assert _nodes_started(store, run_id).count("scorer") == 2


def test_completed_run_cannot_be_resumed(synthetic_bundle, tmp_path):
    store = RunStore(tmp_path)
    final = list(stream_investigation(_with_score(synthetic_bundle, 0.05), "1:A", store=store))[-1][1]
    with pytest.raises(ValueError, match="already completed"):
        list(resume_investigation(synthetic_bundle, final["run_id"], store=store))


def test_guard_replaces_unsafe_llm_draft(synthetic_bundle, tmp_path):
    store = RunStore(tmp_path)
    llm = _FakeLLM(text="The account is guilty of money laundering.")
    final = list(stream_investigation(_with_score(synthetic_bundle, 0.9), "1:A", llm=llm, store=store))[-1][1]
    assert final["writer_mode"] == "template"
    assert final["guard"] == "replaced"
    assert "guilty" not in final["narrative"]
    assert any("guard replaced" in t for t in final["trace"])


def test_guard_passes_safe_llm_draft(synthetic_bundle, tmp_path):
    store = RunStore(tmp_path)
    llm = _FakeLLM(text="Activity shows fan-out to many counterparties, which needs analyst review [KB1].")
    final = list(stream_investigation(_with_score(synthetic_bundle, 0.9), "1:A", llm=llm, store=store))[-1][1]
    assert final["writer_mode"] == "llm"
    assert final["guard"] == "passed"


@pytest.mark.parametrize("text", [
    "They committed fraud on this account.",
    "Customer should be arrested.",
    "Here is how to evade detection by splitting transfers.",
])
def test_guard_patterns_block_unsafe_text(text):
    assert agents.guard_narrative(text) is not None


def test_run_log_is_json_lines(synthetic_bundle, tmp_path):
    store = RunStore(tmp_path)
    final = list(stream_investigation(_with_score(synthetic_bundle, 0.05), "1:A", store=store))[-1][1]
    for line in (tmp_path / f"{final['run_id']}.jsonl").read_text().splitlines():
        assert "ts" in json.loads(line)


# ── Real artifacts: crash every agent, resume, and compare with an uninterrupted run ──

def _real_bundle():
    from aml_copilot.copilot import pipeline
    from aml_copilot.copilot.pipeline import ARTIFACTS

    if not (ARTIFACTS / "model.joblib").exists():
        pytest.skip("bank artifacts not built")
    return pipeline.load_bundle()


@pytest.mark.parametrize("fail_node", ["scorer", "graph_investigator", "retriever", "writer", "decide"])
def test_resumed_run_matches_uninterrupted_run(tmp_path, fail_node):
    bundle = _real_bundle()
    ab = bundle.as_agent_bundle()
    store = RunStore(tmp_path)
    accounts = list(bundle.labels[bundle.labels == 1].index[:8])  # known laundering accounts take the full route
    checked = 0
    for account in accounts:
        base = list(stream_investigation(ab, account, store=store))[-1][1]
        if base["writer_mode"] == "none":
            continue  # low-risk short route: the failing agent is never reached
        before = set(store.list_runs())
        with pytest.raises(RuntimeError, match="injected failure"):
            list(stream_investigation(ab, account, store=store, fail_at=fail_node))
        (run_id,) = set(store.list_runs()) - before
        rec = store.load(run_id)
        assert rec.status == "failed" and rec.failed_node == fail_node

        resumed = list(resume_investigation(ab, run_id, store=store))[-1][1]
        assert store.load(run_id).status == "completed"
        for key in ("score", "disposition", "flags", "graph_summary", "writer_mode", "narrative"):
            assert resumed.get(key) == base.get(key), (account, key)
        checked += 1
    assert checked >= 2, "too few full-route accounts to test this agent"


def test_flagging_follows_threshold(tmp_path):
    from aml_copilot.copilot.model import DECISION_THRESHOLD

    bundle = _real_bundle()
    ab = bundle.as_agent_bundle()
    store = RunStore(tmp_path)
    rows = list(bundle.features.index[:300:10])
    for account in rows:
        final = list(stream_investigation(ab, account, store=store))[-1][1]
        expected = "ESCALATE" if final["score"] >= DECISION_THRESHOLD else "CLEAR"
        assert final["disposition"] == expected, account
