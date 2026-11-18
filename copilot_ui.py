"""AML Investigation Copilot — live Streamlit UI.

Run from the repo root after building artifacts:
    PYTHONPATH=src .venv/bin/streamlit run copilot_ui.py

Pages
-----
Command center   KPIs, score separation, pipeline overview
Network          interactive transaction neighbourhood around any account
Live workflow    watch the LangGraph agents run node by node
Real-time test   stream a batch of investigations and score them against labels
Knowledge base   retrieval over typology passages, with relevance chart
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from aml_copilot.copilot import load_bundle
from aml_copilot.copilot.agents import (
    DONE_KEY,
    LOW_RISK_CUTOFF,
    NODE_ORDER,
    build_app,
    stream_investigation,
)
from aml_copilot.copilot.model import DECISION_THRESHOLD, FEATURES
from aml_copilot.copilot.replay import replay_state, step_summary
from aml_copilot.copilot.viz import (
    confusion_figure,
    ego_network_figure,
    risk_gauge_figure,
    score_distribution_figure,
    workflow_figure,
)

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="AML Investigation Copilot", page_icon="🕵️", layout="wide")

CSS = """
<style>
.hero {
  padding: 1.4rem 1.6rem; border-radius: 16px;
  background: radial-gradient(1200px 400px at 0% 0%, #1e3a8a 0%, #0f172a 55%, #020617 100%);
  border: 1px solid #1e293b; margin-bottom: 1rem;
}
.hero h1 { margin: 0; font-size: 2.0rem; color: #f8fafc; }
.hero p { margin: .35rem 0 0 0; color: #94a3b8; }
.pill {
  display: inline-block; padding: 2px 10px; margin: 6px 6px 0 0; border-radius: 999px;
  background: #0b1220; border: 1px solid #334155; color: #cbd5e1; font-size: 12px;
}
.card {
  padding: .9rem 1rem; border-radius: 14px; background: #0f172a; border: 1px solid #1e293b;
}
.feed-row { font-family: ui-monospace, monospace; font-size: 13px; color: #cbd5e1; }
.badge-esc { color:#fecaca; background:#7f1d1d; padding:4px 12px; border-radius:999px; font-weight:600; }
.badge-clr { color:#bbf7d0; background:#14532d; padding:4px 12px; border-radius:999px; font-weight:600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from .env without overriding existing variables."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value and key not in os.environ:
            os.environ[key] = value


_load_env_file(ROOT / ".env")
API_KEY_PRESENT = bool(os.environ.get("ANTHROPIC_API_KEY"))


@st.cache_resource(show_spinner="Loading graph, model and knowledge base…")
def get_bundle():
    return load_bundle()


@st.cache_data(show_spinner="Scoring all accounts…")
def score_series(_bundle) -> pd.Series:
    proba = _bundle.model.estimator.predict_proba(_bundle.features[FEATURES])[:, 1]
    return pd.Series(proba, index=_bundle.features.index, name="risk_score")


def hero(title: str, subtitle: str, pills: list[str]) -> None:
    chips = "".join(f'<span class="pill">{p}</span>' for p in pills)
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{chips}</div>',
        unsafe_allow_html=True,
    )


bundle = get_bundle()
scores = score_series(bundle)
labels = bundle.labels
ranked_accounts = scores.sort_values(ascending=False).index.tolist()

with st.sidebar:
    st.markdown("### 🕵️ AML Copilot")
    page = st.radio(
        "Navigate",
        ["🏠 Command center", "🕸️ Network", "⚡ Live workflow", "⏯ Playback", "🧪 Real-time test", "📚 Knowledge base"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Accounts: **{len(scores):,}**  ·  Transactions: **{bundle.metrics['n_transactions']:,}**")
    st.caption(f"Claude drafting: **{'available' if API_KEY_PRESENT else 'off (no key)'}**")

if page.startswith("🏠"):
    m = bundle.metrics
    hero(
        "AML Investigation Copilot",
        "Transaction graph · gradient-boosted risk model · RAG over typologies · LangGraph multi-agent workflow",
        ["graph analytics", "ML scoring", "RAG", "multi-agent", "streaming UI"],
    )
    c = st.columns(5)
    c[0].metric("Accounts", f"{m['n_accounts']:,}")
    c[1].metric("Transactions", f"{m['n_transactions']:,}")
    c[2].metric("ROC-AUC", m["roc_auc"], help="Stratified 30% account holdout")
    c[3].metric("PR-AUC", m["pr_auc"])
    c[4].metric("Precision @ top 1%", m["precision_at_top_1pct"])

    left, right = st.columns([3, 2])
    with left:
        st.subheader("How well do scores separate illicit accounts?")
        st.plotly_chart(score_distribution_figure(scores, labels))
    with right:
        st.subheader("Pipeline")
        st.plotly_chart(workflow_figure({n: "pending" for n in NODE_ORDER}))
        st.markdown(
            f"- Escalate at score **≥ {DECISION_THRESHOLD}**\n"
            f"- Clear without investigation below **{LOW_RISK_CUTOFF}**\n"
            f"- Illicit label = account in any laundering transfer (sender or receiver)"
        )
    st.info(
        f"This sample has a {m['illicit_rate']:.1%} illicit-account rate. The full IBM Kaggle data is far more "
        "imbalanced (about 0.1% laundering), so expect lower precision there."
    )

elif page.startswith("🕸️"):
    hero("Transaction network", "Explore the neighbourhood of any account. Node colour is model risk; loops are highlighted in orange.",
         ["plotly", "networkx", "SCC loops"])
    c1, c2, c3 = st.columns([2, 1, 1])
    account = c1.selectbox(
        "Account (highest risk first)", ranked_accounts[:2000],
        format_func=lambda a: f"{a}  ·  risk {scores[a]:.3f}  ·  {'illicit' if labels.get(a, 0) else 'clean'}",
    )
    hops = c2.slider("Hops", 1, 3, 2)
    max_nodes = c3.slider("Max nodes", 20, 200, 80, step=10)

    fig = ego_network_figure(bundle.graph, scores, labels, account, hops=hops, max_nodes=max_nodes)
    st.plotly_chart(fig, key="ego")

    feats = bundle.features.loc[account]
    k = st.columns(5)
    k[0].metric("Risk", f"{scores[account]:.3f}")
    k[1].metric("Outbound partners", int(feats["out_partners"]))
    k[2].metric("Inbound partners", int(feats["in_partners"]))
    k[3].metric("In transaction loop", "yes" if feats["scc_size"] > 1 else "no")
    k[4].metric("Forward ratio", f"{feats['forward_ratio']:.2f}")

    t1, t2 = st.columns(2)
    out_edges = pd.DataFrame(
        [{"counterparty": v, "txns": d["n_txn"], "total": round(d["total"], 2)}
         for _, v, d in bundle.graph.out_edges(account, data=True)]
    ).sort_values("total", ascending=False)
    in_edges = pd.DataFrame(
        [{"counterparty": u, "txns": d["n_txn"], "total": round(d["total"], 2)}
         for u, _, d in bundle.graph.in_edges(account, data=True)]
    ).sort_values("total", ascending=False)
    t1.markdown("**Outbound flows**")
    t1.dataframe(out_edges.head(15), hide_index=True)
    t2.markdown("**Inbound flows**")
    t2.dataframe(in_edges.head(15), hide_index=True)

elif page.startswith("⚡"):
    hero("Live agent workflow", "Pick an account and watch each agent run. Nodes light up as they finish; the trace streams in below.",
         ["LangGraph", "conditional routing", "RAG", "Claude writer"])
    c1, c2 = st.columns([3, 1])
    account = c1.selectbox(
        "Account", ranked_accounts[:2000],
        format_func=lambda a: f"{a}  ·  risk {scores[a]:.3f}  ·  {'illicit' if labels.get(a, 0) else 'clean'}",
        key="live_account",
    )
    use_llm = c2.toggle("Claude writer", value=API_KEY_PRESENT, disabled=not API_KEY_PRESENT)
    run = st.button("▶ Run investigation", type="primary")

    diagram_slot = st.empty()
    progress_slot = st.empty()
    feed_slot = st.container()

    status = {n: "pending" for n in NODE_ORDER}
    detail: dict[str, str] = {}
    if not run and "live_result" not in st.session_state:
        diagram_slot.plotly_chart(workflow_figure(status))
    if run:
        llm = None
        if use_llm:
            import anthropic

            llm = anthropic.Anthropic()
        status["scorer"] = "running"
        diagram_slot.plotly_chart(workflow_figure(status), key=f"wf-{time.time()}")
        progress_slot.caption("Scorer is running…")
        final = {}
        started = time.perf_counter()
        last = started
        with feed_slot:
            feed = st.container()
        for node, update in stream_investigation(bundle.as_agent_bundle(), account, llm=llm):
            now = time.perf_counter()
            if node == DONE_KEY:
                final = update
                break
            status[node] = "done"
            if node == "scorer":
                detail["scorer"] = f"score {update['score']:.3f}"
                if update["score"] < LOW_RISK_CUTOFF:
                    for skipped in ("graph_investigator", "retriever", "writer"):
                        status[skipped] = "skipped"
                    nxt = "decide"
                else:
                    nxt = "graph_investigator"
            elif node == "graph_investigator":
                detail["graph_investigator"] = f"{update['graph_summary']['out_partners']} out · {update['graph_summary']['in_partners']} in"
                nxt = "retriever"
            elif node == "retriever":
                detail["retriever"] = f"{len(update['evidence'])} passages"
                nxt = "writer"
            elif node == "writer":
                detail["writer"] = update.get("writer_mode", "")
                nxt = "decide"
            else:
                nxt = None
            if nxt and status.get(nxt) == "pending":
                status[nxt] = "running"
            for line in update.get("trace", []):
                feed.markdown(f'<div class="feed-row">+{(now - started) * 1000:7.1f} ms &nbsp; {line}</div>',
                              unsafe_allow_html=True)
            diagram_slot.plotly_chart(workflow_figure(status, detail), key=f"wf-{node}-{now}")
            last = now
        status["decide"] = "done"
        total_ms = (time.perf_counter() - started) * 1000
        diagram_slot.plotly_chart(workflow_figure(status, detail), key=f"wf-final-{total_ms}")
        progress_slot.success(f"Completed in {total_ms:.0f} ms")
        st.session_state["live_result"] = final

    result = st.session_state.get("live_result")
    if result and result.get("account_id") == account:
        st.divider()
        g1, g2 = st.columns([1, 2])
        with g1:
            st.plotly_chart(risk_gauge_figure(result["score"]), key="gauge")
            badge = "badge-esc" if result["disposition"] == "ESCALATE" else "badge-clr"
            st.markdown(
                f'<div style="text-align:center"><span class="{badge}">{result["disposition"]}</span>'
                f'&nbsp; writer: <code>{result.get("writer_mode", "n/a")}</code></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Ground truth label: {'illicit' if labels.get(account, 0) else 'clean'}")
        with g2:
            st.subheader("Case narrative")
            st.markdown(result.get("narrative", "_no narrative_"))
            if result.get("flags"):
                st.markdown("**Elevated features**: " + ", ".join(f"`{n}` z={z}" for n, z in result["flags"]))

        if result.get("evidence"):
            st.subheader("Retrieved typology guidance (RAG)")
            st.caption(f"Query: {result.get('query')}")
            for i, ev in enumerate(result["evidence"], start=1):
                with st.expander(f"[KB{i}] {ev['title']}  ·  relevance {ev['score']}"):
                    st.write(ev["text"])

elif page.startswith("⏯"):
    hero("Investigation playback", "Record an investigation once, then scrub or auto-play it step by step. "
         "The diagram and trace rebuild exactly as the agents ran.",
         ["record & replay", "scrub", "auto-play"])
    c1, c2, c3 = st.columns([3, 1, 1])
    account = c1.selectbox(
        "Account", ranked_accounts[:2000],
        format_func=lambda a: f"{a}  ·  risk {scores[a]:.3f}  ·  {'illicit' if labels.get(a, 0) else 'clean'}",
        key="play_account",
    )
    use_llm = c2.toggle("Claude writer", value=False, disabled=not API_KEY_PRESENT, key="play_llm")
    speed = c3.select_slider("Speed", options=[1.5, 1.0, 0.6, 0.3], value=0.6,
                             format_func=lambda s: f"{s}s / step")

    if st.button("⏺ Record investigation", type="primary"):
        llm = None
        if use_llm:
            import anthropic

            llm = anthropic.Anthropic()
        events: list = []
        final: dict = {}
        started = time.perf_counter()
        with st.spinner("Recording agent run…"):
            for node, update in stream_investigation(bundle.as_agent_bundle(), account, llm=llm):
                if node == DONE_KEY:
                    final = update
                    break
                events.append((node, update, (time.perf_counter() - started) * 1000))
        st.session_state["recording"] = {"account": account, "events": events, "final": final}

    rec = st.session_state.get("recording")
    if not rec or rec["account"] != account:
        st.info("Record an investigation for this account to enable playback.")
    else:
        events = rec["events"]
        n = len(events)
        step = st.slider("Step", 0, n, value=n, key=f"play_step_{account}_{n}",
                         help="0 = start, last = finished")
        play = st.button("▶ Auto-play")
        slot_diagram = st.empty()
        slot_trace = st.empty()
        slot_detail = st.empty()

        def render(k: int) -> None:
            status, detail, trace = replay_state(events, k)
            slot_diagram.plotly_chart(workflow_figure(status, detail), key=f"play-{k}-{account}")
            slot_trace.code("\n".join(trace) or "(no steps yet)", language="text")
            if k:
                slot_detail.dataframe(pd.DataFrame([step_summary(events, i) for i in range(k)]),
                                      hide_index=True)
            else:
                slot_detail.empty()

        if play:
            for k in range(1, n + 1):
                render(k)
                time.sleep(speed)
        else:
            render(step)

        final = rec["final"]
        if step == n and final:
            st.success(f"Finished: {final['disposition']} · score {final['score']:.3f} · "
                       f"writer {final.get('writer_mode', 'n/a')}")

elif page.startswith("🧪"):
    hero("Real-time testing", "Stream investigations over a stratified sample and watch the confusion matrix and latency update live.",
         ["batch streaming", "live metrics", "ground-truth check"])
    c1, c2, c3 = st.columns(3)
    n_total = c1.slider("Accounts to test", 10, 300, 80, step=10)
    seed = c2.number_input("Sample seed", value=7, step=1)
    threshold = c3.slider("Escalate threshold", 0.05, 0.95, float(DECISION_THRESHOLD), 0.05,
                          help="Overrides the decision threshold for this test run only")
    run = st.button("▶ Start streaming test", type="primary")

    kpi = st.empty()
    cols = st.columns([1, 1])
    matrix_slot = cols[0].empty()
    latency_slot = cols[1].empty()
    feed_slot = st.empty()

    if run:
        illicit = labels[labels == 1].index.to_series().sample(
            min(n_total // 2, int((labels == 1).sum())), random_state=int(seed)).tolist()
        clean_pool = labels[labels == 0].index.to_series()
        clean = clean_pool.sample(n_total - len(illicit), random_state=int(seed)).tolist()
        sample = pd.Series(illicit + clean).sample(frac=1, random_state=int(seed)).tolist()

        app = build_app(bundle.as_agent_bundle(), llm=None)
        tp = fp = tn = fn = 0
        latencies, rows = [], []
        for i, account in enumerate(sample, start=1):
            t0 = time.perf_counter()
            out = app.invoke({"account_id": account, "trace": []})
            latencies.append((time.perf_counter() - t0) * 1000)
            score = out["score"]
            predicted = score >= threshold
            actual = bool(labels.get(account, 0))
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and actual:
                fn += 1
            else:
                tn += 1
            rows.append({
                "#": i, "account": account, "score": round(score, 3),
                "decision": "ESCALATE" if predicted else "CLEAR",
                "truth": "illicit" if actual else "clean",
                "correct": "✅" if predicted == actual else "❌",
                "writer": out.get("writer_mode", "none"),
                "ms": round(latencies[-1], 1),
            })
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            acc = (tp + tn) / i
            with kpi.container():
                k = st.columns(5)
                k[0].metric("Processed", f"{i}/{len(sample)}")
                k[1].metric("Accuracy", f"{acc:.1%}")
                k[2].metric("Precision", f"{prec:.1%}")
                k[3].metric("Recall", f"{rec:.1%}")
                k[4].metric("p50 latency", f"{sorted(latencies)[len(latencies)//2]:.1f} ms")
            if i % 3 == 0 or i == len(sample):
                matrix_slot.plotly_chart(confusion_figure(tp, fp, tn, fn), key=f"cm-{i}")
                latency_slot.plotly_chart(
                    go.Figure(go.Scatter(y=latencies, mode="lines+markers",
                                         line=dict(color="#60a5fa"))).update_layout(
                        height=280, title="Latency per investigation (ms)",
                        margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"),
                    key=f"lat-{i}")
                feed_slot.dataframe(pd.DataFrame(rows[::-1][:15]), hide_index=True)
        st.success(f"Done. {len(sample)} investigations, threshold {threshold:.2f}. Writer: template (no API calls).")

elif page.startswith("📚"):
    hero("Typology knowledge base", "The passages the retriever agent draws on, ranked by relevance to your query.",
         ["TF-IDF", "offline", "RAG"])
    kb = bundle.kb
    query = st.text_input("Ask the knowledge base", value="many small transfers to many counterparties")
    hits = kb.search(query, k=8)
    if hits:
        chart = go.Figure(go.Bar(
            x=[h["score"] for h in hits][::-1], y=[h["title"][:50] for h in hits][::-1],
            orientation="h", marker=dict(color=[h["score"] for h in hits][::-1], colorscale="Blues"),
        ))
        chart.update_layout(height=max(220, 40 * len(hits)), margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="relevance", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(chart, key="kb-bar")
        for h in hits:
            with st.expander(f"{h['title']}  ·  {h['source']}  ·  {h['score']}"):
                st.write(h["text"])
    else:
        st.info("No matching passages. Try words like fan-out, layering, loop, mule.")
    st.caption(f"{len(kb.chunks)} passages indexed from docs/kb/")
