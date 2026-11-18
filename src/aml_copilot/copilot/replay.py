"""Turn a recorded investigation into per-step workflow state for playback.

A recording is a list of ``(node, update, t_ms)`` tuples, as produced by
``stream_investigation`` with timings attached. ``replay_state`` reconstructs what the
diagram and trace should show after the first ``upto`` events, so the same logic can drive
both live rendering and step-by-step replay.
"""
from __future__ import annotations

from typing import Any

from .agents import LOW_RISK_CUTOFF, NODE_ORDER

Event = tuple[str, dict[str, Any], float]


def _next_node(node: str, update: dict[str, Any]) -> str | None:
    if node == "scorer":
        return "decide" if update["score"] < LOW_RISK_CUTOFF else "graph_investigator"
    return {
        "graph_investigator": "retriever",
        "retriever": "writer",
        "writer": "decide",
    }.get(node)


def replay_state(events: list[Event], upto: int) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Node statuses, short per-node details and trace lines after the first ``upto`` events."""
    status = {n: "pending" for n in NODE_ORDER}
    detail: dict[str, str] = {}
    trace: list[str] = []
    upto = max(0, min(upto, len(events)))
    if upto == 0:
        return status, detail, trace

    status["scorer"] = "running"
    for node, update, t_ms in events[:upto]:
        status[node] = "done"
        if node == "scorer":
            detail["scorer"] = f"score {update['score']:.3f}"
            if update["score"] < LOW_RISK_CUTOFF:
                for skipped in ("graph_investigator", "retriever", "writer"):
                    status[skipped] = "skipped"
        elif node == "graph_investigator":
            summary = update["graph_summary"]
            detail[node] = f"{summary['out_partners']} out · {summary['in_partners']} in"
        elif node == "retriever":
            detail[node] = f"{len(update['evidence'])} passages ({update.get('retrieval_mode', '?')})"
        elif node == "writer":
            detail[node] = update.get("writer_mode", "")
        for line in update.get("trace", []):
            trace.append(f"+{t_ms:8.1f} ms  {line}")

        nxt = _next_node(node, update)
        if nxt and status.get(nxt) == "pending":
            status[nxt] = "running"
    return status, detail, trace


def step_summary(events: list[Event], index: int) -> dict[str, Any]:
    """Compact description of event ``index`` (0-based) for a timeline table."""
    node, update, t_ms = events[index]
    keys = {
        "scorer": lambda u: {"score": round(u["score"], 3), "elevated": len(u["flags"])},
        "graph_investigator": lambda u: {"out_partners": u["graph_summary"]["out_partners"],
                                         "in_cycle": u["graph_summary"]["in_cycle"]},
        "retriever": lambda u: {"passages": len(u["evidence"]), "mode": u.get("retrieval_mode")},
        "writer": lambda u: {"writer": u.get("writer_mode")},
        "decide": lambda u: {"disposition": u.get("disposition")},
    }
    return {"step": index + 1, "agent": node, "t_ms": round(t_ms, 1),
            **(keys[node](update) if node in keys else {})}
