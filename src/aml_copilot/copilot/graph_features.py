"""Transaction-graph construction and per-account structural features.

The graph is directed: an edge src -> dst per (src, dst) pair, aggregated over
all transfers between them. Features are unsupervised (they never read labels),
so computing them on the full graph does not leak the target.
"""
from __future__ import annotations

import bisect

import networkx as nx
import pandas as pd

FORWARD_WINDOW_S = 24 * 3600  # an inbound amount is "forwarded" within 24h
FORWARD_TOLERANCE = 0.2  # outbound amount within 20% of the inbound amount


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    agg = (
        df.groupby(["src", "dst"])
        .agg(n_txn=("amount_paid", "size"), total=("amount_paid", "sum"))
        .reset_index()
    )
    graph = nx.DiGraph()
    graph.add_edges_from(
        (r.src, r.dst, {"n_txn": int(r.n_txn), "total": float(r.total)})
        for r in agg.itertuples(index=False)
    )
    return graph


def _forward_ratio(
    inbound: list[tuple[float, float]], outbound: list[tuple[float, float]]
) -> float:
    """Share of inbound transfers that were passed on quickly in similar size."""
    if not inbound or not outbound:
        return 0.0
    out_times = [t for t, _ in outbound]
    hits = 0
    for t_in, amt_in in inbound:
        if amt_in <= 0:
            continue
        start = bisect.bisect_left(out_times, t_in)
        for t_out, amt_out in outbound[start:]:
            if t_out - t_in > FORWARD_WINDOW_S:
                break
            if abs(amt_out - amt_in) <= FORWARD_TOLERANCE * amt_in:
                hits += 1
                break
    return hits / len(inbound)


def account_features(df: pd.DataFrame, graph: nx.DiGraph | None = None) -> pd.DataFrame:
    """One row per account with structural and behavioural features."""
    if graph is None:
        graph = build_graph(df)

    t0 = df["timestamp"].min()
    times = (df["timestamp"] - t0).dt.total_seconds()
    work = df.assign(ts=times)

    out_g = work.groupby("src")
    in_g = work.groupby("dst")
    feats = pd.DataFrame(index=pd.Index(sorted(graph.nodes()), name="account"))

    feats["out_txn_count"] = out_g.size().reindex(feats.index, fill_value=0)
    feats["in_txn_count"] = in_g.size().reindex(feats.index, fill_value=0)
    feats["out_total"] = out_g["amount_paid"].sum().reindex(feats.index, fill_value=0.0)
    feats["in_total"] = in_g["amount_received"].sum().reindex(feats.index, fill_value=0.0)
    feats["out_partners"] = out_g["dst"].nunique().reindex(feats.index, fill_value=0)
    feats["in_partners"] = in_g["src"].nunique().reindex(feats.index, fill_value=0)
    feats["mean_out_amount"] = feats["out_total"] / feats["out_txn_count"].clip(lower=1)
    feats["mean_in_amount"] = feats["in_total"] / feats["in_txn_count"].clip(lower=1)

    # Fan-out / fan-in: many distinct partners relative to transaction count.
    feats["partner_diversity_out"] = feats["out_partners"] / feats["out_txn_count"].clip(lower=1)
    feats["partner_diversity_in"] = feats["in_partners"] / feats["in_txn_count"].clip(lower=1)

    # Pass-through: money in roughly equals money out.
    hi = feats[["in_total", "out_total"]].max(axis=1).clip(lower=1e-9)
    lo = feats[["in_total", "out_total"]].min(axis=1)
    feats["in_out_balance"] = lo / hi

    # Reciprocity: share of outgoing partners that also send money back.
    out_sets = {n: set(graph.successors(n)) for n in graph.nodes()}
    in_sets = {n: set(graph.predecessors(n)) for n in graph.nodes()}
    feats["reciprocity"] = [
        len(out_sets[n] & in_sets[n]) / len(out_sets[n]) if out_sets[n] else 0.0
        for n in feats.index
    ]

    # Cycles: members of a non-trivial strongly connected component.
    scc_size = {}
    for comp in nx.strongly_connected_components(graph):
        if len(comp) > 1:
            for node in comp:
                scc_size[node] = len(comp)
    feats["scc_size"] = [scc_size.get(n, 0) for n in feats.index]

    pagerank = nx.pagerank(graph, weight=None)
    feats["pagerank"] = [pagerank.get(n, 0.0) for n in feats.index]

    # Rapid forwarding: inbound funds passed on within 24h at similar size.
    outbound = {
        k: list(zip(g["ts"], g["amount_paid"])) for k, g in work.groupby("src")
    }
    inbound = {
        k: list(zip(g["ts"], g["amount_received"])) for k, g in work.groupby("dst")
    }
    feats["forward_ratio"] = [
        _forward_ratio(inbound.get(n, []), outbound.get(n, [])) for n in feats.index
    ]

    active = pd.concat(
        [out_g["timestamp"].agg(["min", "max"]), in_g["timestamp"].agg(["min", "max"])]
    ).groupby(level=0).agg({"min": "min", "max": "max"})
    span = (active["max"] - active["min"]).dt.total_seconds() / 86400.0
    feats["active_span_days"] = span.reindex(feats.index).fillna(0.0)

    return feats.fillna(0.0).astype(float)
