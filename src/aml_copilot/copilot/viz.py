"""Plotly figure builders for the investigation UI (kept free of Streamlit so they can be tested)."""
from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from .agents import LOW_RISK_CUTOFF
from .model import DECISION_THRESHOLD

PALETTE = {
    "pending": "#3b4252",
    "running": "#fbbf24",
    "done": "#34d399",
    "skipped": "#4b5563",
    "escalate": "#f87171",
    "clear": "#34d399",
    "accent": "#60a5fa",
}

WORKFLOW_POS = {
    "scorer": (0.0, 0.0),
    "graph_investigator": (1.0, 1.0),
    "retriever": (2.0, 1.0),
    "writer": (3.0, 1.0),
    "decide": (4.0, 0.0),
}
WORKFLOW_LABEL = {
    "scorer": "Scorer\nrisk model",
    "graph_investigator": "Graph\ninvestigator",
    "retriever": "Retriever\nRAG",
    "writer": "Writer\nLLM / template",
    "decide": "Decide\nESCALATE / CLEAR",
}
WORKFLOW_EDGES = [
    ("scorer", "graph_investigator", "score ≥ 0.15"),
    ("graph_investigator", "retriever", ""),
    ("retriever", "writer", ""),
    ("writer", "decide", ""),
    ("scorer", "decide", "score < 0.15"),
]

_STATUS_COLOR = {
    "pending": PALETTE["pending"],
    "running": PALETTE["running"],
    "done": PALETTE["done"],
    "skipped": PALETTE["skipped"],
}


def workflow_figure(status: dict[str, str], detail: dict[str, str] | None = None) -> go.Figure:
    """Pipeline diagram where each node is coloured by its live status."""
    detail = detail or {}
    fig = go.Figure()

    for src, dst, label in WORKFLOW_EDGES:
        (x0, y0), (x1, y1) = WORKFLOW_POS[src], WORKFLOW_POS[dst]
        if src == "scorer" and dst == "decide":
            # Short route: taken only when the investigation branch was skipped.
            taken = status.get("graph_investigator") == "skipped" and status.get("decide") == "done"
        else:
            taken = status.get(src) == "done" and status.get(dst) in {"done", "running"}
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2.2 if taken else 1.2,
            arrowcolor=PALETTE["accent"] if taken else "#475569", text="",
        )
        if label:
            fig.add_annotation(
                x=(x0 + x1) / 2, y=(y0 + y1) / 2 + (0.28 if dst != "decide" else -0.3),
                text=label, showarrow=False, font=dict(size=11, color="#94a3b8"),
            )

    xs, ys, colors, texts, hovers = [], [], [], [], []
    for node, (x, y) in WORKFLOW_POS.items():
        st = status.get(node, "pending")
        xs.append(x)
        ys.append(y)
        colors.append(_STATUS_COLOR.get(st, PALETTE["pending"]))
        texts.append(WORKFLOW_LABEL[node])
        hovers.append(f"<b>{node}</b><br>status: {st}<br>{detail.get(node, '')}")

    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="markers+text", text=texts, textposition="bottom center",
            textfont=dict(color="#e2e8f0", size=12),
            marker=dict(size=58, color=colors, line=dict(width=2, color="#0f172a")),
            hovertext=hovers, hoverinfo="text",
        )
    )
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        xaxis=dict(visible=False, range=[-0.6, 4.6]),
        yaxis=dict(visible=False, range=[-0.8, 1.6]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def ego_network_figure(
    graph: nx.DiGraph,
    scores: pd.Series,
    labels: pd.Series,
    center: str,
    hops: int = 2,
    max_nodes: int = 80,
) -> go.Figure:
    """Interactive neighbourhood: node colour = risk score. Links take the lower of their two ends' scores:
    red at or above the decision threshold, amber above the low-risk cutoff, orange for loops."""
    undirected = graph.to_undirected(as_view=True)
    reach = nx.single_source_shortest_path_length(undirected, center, cutoff=hops)
    if len(reach) > max_nodes:
        ranked = sorted(reach, key=lambda n: (reach[n], -undirected.degree(n)))
        keep = set(ranked[:max_nodes])
    else:
        keep = set(reach)
    sub = graph.subgraph(keep).copy()
    pos = nx.spring_layout(sub.to_undirected(), seed=42, k=None)

    cycle_nodes: set[str] = set()
    for comp in nx.strongly_connected_components(sub):
        if len(comp) > 1:
            cycle_nodes |= comp

    # A link is only as risky as its safer end: use the lower of the two scores.
    def link_risk(u: str, v: str) -> float:
        return min(float(scores.get(u, 0.0)), float(scores.get(v, 0.0)))

    bands = {"risky": ([], []), "watched": ([], []), "other": ([], [])}  # loops are drawn as an overlay
    for u, v in sub.edges():
        (x0, y0), (x1, y1) = pos[u], pos[v]
        risk = link_risk(u, v)
        if risk >= DECISION_THRESHOLD:
            key = "risky"
        elif risk >= LOW_RISK_CUTOFF:
            key = "watched"
        else:
            key = "other"
        bands[key][0].extend([x0, x1, None])
        bands[key][1].extend([y0, y1, None])
    fig = go.Figure()
    style = {
        "other": dict(width=1, color="#475569", name="transfers"),
        "watched": dict(width=2.5, color="#fbbf24", name="watched link (risk 0.15–0.5)"),
        "risky": dict(width=4, color="#f87171", name="risky link (risk ≥ 0.5 at both ends)"),
    }
    for key in ("other", "watched", "risky"):
        xs_e, ys_e = bands[key]
        if xs_e:
            fig.add_trace(go.Scatter(x=xs_e, y=ys_e, mode="lines", line=dict(
                width=style[key]["width"], color=style[key]["color"]),
                hoverinfo="skip", name=style[key]["name"]))

    # Loops are drawn on top of the risk colours, so a risky loop still shows as a loop.
    loop_x, loop_y = [], []
    for u, v in sub.edges():
        if u in cycle_nodes and v in cycle_nodes:
            (x0, y0), (x1, y1) = pos[u], pos[v]
            loop_x += [x0, x1, None]
            loop_y += [y0, y1, None]
    if loop_x:
        fig.add_trace(go.Scatter(x=loop_x, y=loop_y, mode="lines",
                                 line=dict(width=3, color="#fb923c", dash="dot"),
                                 hoverinfo="skip", name="loop (money returns to its source)"))

    nodes = list(sub.nodes())
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    deg = dict(sub.degree())
    max_deg = max(deg.values()) if deg else 1
    sizes = [12 + 26 * deg[n] / max_deg for n in nodes]
    colors = [float(scores.get(n, 0.0)) for n in nodes]
    symbols = ["star" if n == center else ("diamond" if n in cycle_nodes else "circle") for n in nodes]
    hovers = [
        f"<b>{n}</b><br>risk score: {scores.get(n, 0.0):.3f}<br>"
        f"illicit label: {int(labels.get(n, 0))}<br>degree: {deg[n]}"
        + ("<br><i>in transaction loop</i>" if n in cycle_nodes else "")
        for n in nodes
    ]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        text=[n.split(":")[-1] if len(nodes) <= 40 else "" for n in nodes],
        textposition="top center", textfont=dict(size=10, color="#e2e8f0"),
        marker=dict(size=sizes, color=colors, colorscale="Turbo", cmin=0, cmax=1,
                    symbol=symbols, line=dict(width=1, color="#0f172a"),
                    colorbar=dict(title="risk", thickness=12)),
        hovertext=hovers, hoverinfo="text", name="accounts",
    ))
    fig.update_layout(
        height=560, margin=dict(l=0, r=0, t=10, b=0), showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def score_distribution_figure(scores: pd.Series, labels: pd.Series) -> go.Figure:
    aligned = pd.DataFrame({"score": scores, "label": labels.reindex(scores.index).fillna(0).astype(int)})
    fig = go.Figure()
    for lab, name, color in [(0, "clean", PALETTE["done"]), (1, "illicit", PALETTE["escalate"])]:
        vals = aligned.loc[aligned["label"] == lab, "score"]
        fig.add_trace(go.Histogram(x=vals, name=f"{name} ({len(vals):,})", nbinsx=40,
                                   marker_color=color, opacity=0.65))
    fig.add_vline(x=0.15, line_dash="dot", line_color="#94a3b8", annotation_text="clear cutoff")
    fig.add_vline(x=0.5, line_dash="dash", line_color="#94a3b8", annotation_text="escalate")
    fig.update_layout(
        barmode="overlay", height=340, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="model risk score", yaxis_title="accounts",
        legend=dict(orientation="h", y=1.1),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def risk_gauge_figure(score: float) -> go.Figure:
    color = PALETTE["escalate"] if score >= 0.5 else (PALETTE["running"] if score >= 0.15 else PALETTE["clear"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number=dict(valueformat=".3f", font=dict(size=34)),
        gauge=dict(
            axis=dict(range=[0, 1]),
            bar=dict(color=color),
            steps=[dict(range=[0, 0.15], color="#14532d"),
                   dict(range=[0.15, 0.5], color="#713f12"),
                   dict(range=[0.5, 1], color="#7f1d1d")],
            threshold=dict(line=dict(color="white", width=3), value=0.5),
        ),
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def confusion_figure(tp: int, fp: int, tn: int, fn: int) -> go.Figure:
    z = [[tn, fp], [fn, tp]]
    text = [[f"TN<br>{tn}", f"FP<br>{fp}"], [f"FN<br>{fn}", f"TP<br>{tp}"]]
    fig = go.Figure(go.Heatmap(
        z=z, text=text, texttemplate="%{text}", textfont=dict(size=16),
        x=["predicted clean", "predicted illicit"], y=["actual clean", "actual illicit"],
        colorscale=[[0, "#1e293b"], [1, PALETTE["accent"]]], showscale=False,
        hoverinfo="skip",
    ))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(autorange="reversed")
    return fig
