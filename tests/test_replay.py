"""Tests for investigation playback state. Offline, uses hand-written recordings."""
from __future__ import annotations

from aml_copilot.copilot.replay import replay_state, step_summary


def _full_route():
    return [
        ("scorer", {"score": 0.9, "flags": [("out_partners", 4.0)], "trace": ["scorer: risk=0.900"]}, 1.0),
        ("graph_investigator", {"graph_summary": {"out_partners": 7, "in_partners": 2, "in_cycle": True},
                                "trace": ["graph: out=7"]}, 5.0),
        ("retriever", {"evidence": [{}, {}], "retrieval_mode": "graph", "trace": ["retriever: graph"]}, 9.0),
        ("writer", {"writer_mode": "template", "narrative": "x", "trace": ["writer: template"]}, 12.0),
        ("decide", {"disposition": "ESCALATE", "trace": ["decide: ESCALATE"]}, 13.0),
    ]


def _short_route():
    return [
        ("scorer", {"score": 0.02, "flags": [], "trace": ["scorer: risk=0.020"]}, 1.0),
        ("decide", {"disposition": "CLEAR", "trace": ["decide: CLEAR"]}, 2.0),
    ]


def test_nothing_played_yet_is_all_pending():
    status, detail, trace = replay_state(_full_route(), 0)
    assert set(status.values()) == {"pending"}
    assert detail == {} and trace == []


def test_mid_run_shows_done_running_and_pending():
    status, detail, trace = replay_state(_full_route(), 2)
    assert status["scorer"] == "done"
    assert status["graph_investigator"] == "done"
    assert status["retriever"] == "running"
    assert status["writer"] == "pending"
    assert "retriever" not in detail  # retriever has not produced output yet
    assert len(trace) == 2
    _, detail_after, _ = replay_state(_full_route(), 3)
    assert detail_after["retriever"] == "2 passages (graph)"


def test_finished_run_marks_every_node_done():
    status, _, trace = replay_state(_full_route(), 5)
    assert all(s == "done" for s in status.values())
    assert trace[-1].endswith("decide: ESCALATE")


def test_short_route_skips_investigation_nodes():
    status, _, _ = replay_state(_short_route(), 1)
    assert status["graph_investigator"] == "skipped"
    assert status["retriever"] == "skipped"
    assert status["writer"] == "skipped"
    assert status["decide"] == "running"
    status_end, _, _ = replay_state(_short_route(), 2)
    assert status_end["decide"] == "done"


def test_upto_is_clamped():
    assert replay_state(_full_route(), 99)[0] == replay_state(_full_route(), 5)[0]
    assert replay_state(_full_route(), -3)[0] == replay_state(_full_route(), 0)[0]


def test_step_summary_reports_agent_fields():
    row = step_summary(_full_route(), 0)
    assert row == {"step": 1, "agent": "scorer", "t_ms": 1.0, "score": 0.9, "elevated": 1}
    assert step_summary(_full_route(), 2)["mode"] == "graph"
