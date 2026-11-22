"""Multi-agent investigation workflow (LangGraph).

Specialist agents share one case state:

    scorer ──(score < cutoff)──────────────────────────────┐
       │                                                    │
       ▼                                                    │
    graph_investigator → retriever → writer ──────────────► decide → END

- scorer: runs the risk model and finds elevated features.
- graph_investigator: summarizes the account's graph neighbourhood.
- retriever: pulls typology passages from the knowledge base (RAG).
- writer: drafts the narrative with Claude when a client is supplied,
  otherwise with a deterministic template (works offline).
- decide: maps the score to ESCALATE or CLEAR.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

import networkx as nx
from langgraph.graph import END, START, StateGraph

from .model import DECISION_THRESHOLD, ScoringModel
from .rag import KnowledgeBase
from .rag_graph import GraphRetriever, signals_from_case

LOW_RISK_CUTOFF = 0.15  # below this the case is cleared without investigation
WRITER_MODEL = "claude-haiku-5-5"

FLAG_QUERIES = {
    "out_partners": "fan-out dispersal many distinct counterparties smurfing",
    "partner_diversity_out": "fan-out dispersal many distinct counterparties",
    "in_partners": "fan-in collection many senders consolidation",
    "partner_diversity_in": "fan-in collection many senders",
    "in_out_balance": "pass-through balanced inflow outflow layering",
    "forward_ratio": "rapid forwarding pass-through layering within 24 hours",
    "scc_size": "circular flows cycles round-tripping loop",
    "reciprocity": "reciprocity round-tripping two-way traffic",
    "pagerank": "central hub account network",
    "active_span_days": "new account sudden spike money mule",
    "out_total": "large outbound transfers",
    "in_total": "large inbound transfers",
}


class CaseState(TypedDict, total=False):
    account_id: str
    score: float
    flags: list[tuple[str, float]]
    graph_summary: dict[str, Any]
    query: str
    evidence: list[dict[str, Any]]
    retrieval_mode: str
    narrative: str
    writer_mode: str
    disposition: str
    trace: Annotated[list[str], operator.add]


class AgentBundle(TypedDict, total=False):
    model: ScoringModel
    features: Any  # pandas.DataFrame indexed by account
    graph: nx.DiGraph
    kb: KnowledgeBase
    graph_rag: GraphRetriever | None  # optional; TF-IDF retrieval is used when absent


def _graph_summary(graph: nx.DiGraph, features, account: str, top_n: int = 3) -> dict[str, Any]:
    outs = sorted(
        ((dst, d["total"], d["n_txn"]) for _, dst, d in graph.out_edges(account, data=True)),
        key=lambda x: -x[1],
    )[:top_n]
    ins = sorted(
        ((src, d["total"], d["n_txn"]) for src, _, d in graph.in_edges(account, data=True)),
        key=lambda x: -x[1],
    )[:top_n]
    reach2 = len(nx.single_source_shortest_path_length(graph, account, cutoff=2)) - 1
    row = features.loc[account]
    return {
        "out_partners": int(row["out_partners"]),
        "in_partners": int(row["in_partners"]),
        "in_cycle": bool(row["scc_size"] > 1),
        "scc_size": int(row["scc_size"]),
        "reciprocity": round(float(row["reciprocity"]), 3),
        "forward_ratio": round(float(row["forward_ratio"]), 3),
        "two_hop_reach": reach2,
        "top_outbound": [
            {"counterparty": c, "total": round(t, 2), "txns": n} for c, t, n in outs
        ],
        "top_inbound": [
            {"counterparty": c, "total": round(t, 2), "txns": n} for c, t, n in ins
        ],
    }


def _template_narrative(state: CaseState) -> str:
    lines = [
        f"Account {state['account_id']} received a model risk score of {state['score']:.2f} "
        f"(decision threshold {DECISION_THRESHOLD})."
    ]
    if state.get("flags"):
        described = ", ".join(f"{name} (z={z})" for name, z in state["flags"])
        lines.append(f"Elevated structural features versus the population: {described}.")
    summary = state.get("graph_summary") or {}
    if summary:
        lines.append(
            f"Graph view: {summary['out_partners']} outbound and {summary['in_partners']} "
            f"inbound counterparties; "
            f"{'inside a transaction loop' if summary['in_cycle'] else 'not inside a transaction loop'}; "
            f"{summary['two_hop_reach']} accounts reachable within two hops."
        )
    for i, ev in enumerate(state.get("evidence") or [], start=1):
        lines.append(f"Relevant guidance [KB{i}] {ev['title']}: {ev['text'][:240]}...")
    lines.append(
        "This draft is generated from fixed rules and is for analyst review only. "
        "It does not establish that a crime occurred."
    )
    return "\n".join(lines)


def _llm_narrative(llm: Any, state: CaseState) -> str:
    facts = {
        "account_id": state["account_id"],
        "score": round(state["score"], 3),
        "elevated_features": state.get("flags", []),
        "graph_summary": state.get("graph_summary", {}),
    }
    evidence = [
        {"tag": f"KB{i}", "title": ev["title"], "text": ev["text"]}
        for i, ev in enumerate(state.get("evidence") or [], start=1)
    ]
    system = (
        "You write AML case narratives for a human analyst. Use only the facts provided. "
        "Cite guidance with its tag, for example [KB1]. Describe suspicious behaviour and the "
        "reasons for review; do not state that a crime occurred. Keep it under 180 words."
    )
    user = f"Facts: {facts}\n\nGuidance: {evidence}"
    response = llm.messages.create(
        model=WRITER_MODEL,
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()


def build_app(bundle: AgentBundle, llm: Any | None = None):
    """Compile the workflow. Pass an Anthropic client as ``llm`` to enable Claude drafting."""
    model, features, graph, kb = bundle["model"], bundle["features"], bundle["graph"], bundle["kb"]
    graph_rag = bundle.get("graph_rag")

    def scorer(state: CaseState) -> dict:
        account = state["account_id"]
        if account not in features.index:
            raise KeyError(f"unknown account: {account}")
        row = features.loc[[account]]
        score = model.predict_proba(row)
        flags = model.elevated_features(row)
        return {
            "score": score,
            "flags": flags,
            "trace": [f"scorer: risk={score:.3f}, elevated_features={len(flags)}"],
        }

    def route_after_score(state: CaseState) -> str:
        return "investigate" if state["score"] >= LOW_RISK_CUTOFF else "clear"

    def graph_investigator(state: CaseState) -> dict:
        summary = _graph_summary(graph, features, state["account_id"])
        return {
            "graph_summary": summary,
            "trace": [
                f"graph_investigator: out={summary['out_partners']} in={summary['in_partners']} "
                f"cycle={summary['in_cycle']}"
            ],
        }

    def retriever(state: CaseState) -> dict:
        terms = [FLAG_QUERIES[name] for name, _ in state.get("flags", []) if name in FLAG_QUERIES]
        summary = state.get("graph_summary", {})
        if summary.get("in_cycle"):
            terms.append(FLAG_QUERIES["scc_size"])
        if summary.get("forward_ratio", 0) > 0.2:
            terms.append(FLAG_QUERIES["forward_ratio"])
        if summary.get("out_partners", 0) >= 5:
            terms.append(FLAG_QUERIES["out_partners"])
        if summary.get("in_partners", 0) >= 5:
            terms.append(FLAG_QUERIES["in_partners"])
        if summary.get("reciprocity", 0) > 0.3:
            terms.append(FLAG_QUERIES["reciprocity"])
        query = " ".join(terms) or "suspicious transaction pattern review"
        if graph_rag is not None:
            signals = signals_from_case(state.get("flags"), state.get("graph_summary"))
            evidence = graph_rag.retrieve(query, signals, k=3)
            mode = "graph"
            trace = f"retriever: graph RAG, {len(evidence)} passages, concepts={sorted(signals)}"
        else:
            evidence = kb.search(query, k=3)
            mode = "tfidf"
            trace = f"retriever: tf-idf, {len(evidence)} passages"
        return {
            "query": query,
            "evidence": evidence,
            "retrieval_mode": mode,
            "trace": [trace],
        }

    def writer(state: CaseState) -> dict:
        if llm is not None:
            try:
                return {
                    "narrative": _llm_narrative(llm, state),
                    "writer_mode": "llm",
                    "trace": [f"writer: llm ({WRITER_MODEL})"],
                }
            except Exception as exc:  # noqa: BLE001 - fall back rather than fail the case
                return {
                    "narrative": _template_narrative(state),
                    "writer_mode": "template",
                    "trace": [f"writer: llm failed ({type(exc).__name__}), used template"],
                }
        return {
            "narrative": _template_narrative(state),
            "writer_mode": "template",
            "trace": ["writer: template"],
        }

    def decide(state: CaseState) -> dict:
        score = state["score"]
        disposition = "ESCALATE" if score >= DECISION_THRESHOLD else "CLEAR"
        update: dict[str, Any] = {"disposition": disposition, "trace": [f"decide: {disposition}"]}
        if "narrative" not in state:
            update["narrative"] = (
                f"Model risk score {score:.2f} is below the investigation cutoff "
                f"({LOW_RISK_CUTOFF}); no further investigation."
            )
            update["writer_mode"] = "none"
        return update

    g = StateGraph(CaseState)
    g.add_node("scorer", scorer)
    g.add_node("graph_investigator", graph_investigator)
    g.add_node("retriever", retriever)
    g.add_node("writer", writer)
    g.add_node("decide", decide)
    g.add_edge(START, "scorer")
    g.add_conditional_edges(
        "scorer", route_after_score, {"investigate": "graph_investigator", "clear": "decide"}
    )
    g.add_edge("graph_investigator", "retriever")
    g.add_edge("retriever", "writer")
    g.add_edge("writer", "decide")
    g.add_edge("decide", END)
    return g.compile()


def investigate(bundle: AgentBundle, account_id: str, llm: Any | None = None) -> dict[str, Any]:
    app = build_app(bundle, llm=llm)
    final = app.invoke({"account_id": account_id, "trace": []})
    return dict(final)


NODE_ORDER = ["scorer", "graph_investigator", "retriever", "writer", "decide"]
DONE_KEY = "__done__"


def stream_investigation(bundle: AgentBundle, account_id: str, llm: Any | None = None):
    """Yield ``(node, update)`` as each agent finishes, then ``(DONE_KEY, final_state)``.

    Lets a UI show the workflow progressing in real time instead of waiting for the end.
    """
    app = build_app(bundle, llm=llm)
    final: dict[str, Any] = {"account_id": account_id, "trace": []}
    for event in app.stream({"account_id": account_id, "trace": []}, stream_mode="updates"):
        for node, update in event.items():
            for key, value in update.items():
                if key == "trace":
                    final["trace"] = final.get("trace", []) + list(value)
                else:
                    final[key] = value
            yield node, update
    yield DONE_KEY, final
