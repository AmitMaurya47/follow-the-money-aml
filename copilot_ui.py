"""AML Investigation Lab — live Streamlit UI.

Run from the repo root after building artifacts:
    PYTHONPATH=src .venv/bin/streamlit run copilot_ui.py

Pages
-----
Welcome          what the tool is, how it works, flowchart, demo sign-in entry
My accounts      demo sign-in; check whether linked accounts were used for laundering
System live      the whole usual process run live, stage by stage, as a story
Command center   KPIs, score separation, pipeline overview
Network          interactive transaction neighbourhood around any account
Live workflow    watch the LangGraph agents run node by node
Playback         record an investigation and replay it step by step
Real-time test   stream a batch of investigations and score them against labels
Knowledge base   retrieval over typology passages, with relevance chart
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from aml_copilot.auth import check_login
from aml_copilot.copilot import landing, load_bundle
from aml_copilot.copilot.agents import (
    DONE_KEY,
    LOW_RISK_CUTOFF,
    NODE_ORDER,
    build_app,
    investigate,
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
st.set_page_config(page_title="Follow the Money", page_icon="🔎", layout="wide")

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
st.markdown(CSS + landing.STYLE, unsafe_allow_html=True)


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

PAGE_NAMES = [
    "🌐 Welcome", "👤 My accounts", "🛰️ System live", "🏠 Command center", "🕸️ Network",
    "⚡ Live workflow", "⏯ Playback", "🧪 Real-time test", "📚 Knowledge base",
    "₿ Crypto (Elliptic)", "💬 Ask the assistant",
]


def go_to(target: str) -> None:
    """Button callback: switch the sidebar page before the radio is drawn."""
    st.session_state["nav"] = target


def sign_out() -> None:
    st.session_state["signed_in"] = False
    st.session_state.pop("user", None)


def demo_accounts(labels: pd.Series) -> list[str]:
    """Fixed demo profile: three accounts that were used for laundering, two that were not."""
    illicit = labels[labels == 1].index.to_series().sample(3, random_state=11).tolist()
    clean = labels[labels == 0].index.to_series().sample(2, random_state=11).tolist()
    return illicit + clean


def status_for(result: dict) -> dict[str, str]:
    """Node statuses for a finished investigation, for the workflow diagram."""
    skipped = result.get("writer_mode") == "none"
    status = {n: "done" for n in NODE_ORDER}
    if skipped:
        for n in ("graph_investigator", "retriever", "writer"):
            status[n] = "skipped"
    return status


with st.sidebar:
    st.markdown("### 🔎 Follow the Money")
    page = st.radio("Navigate", PAGE_NAMES, key="nav", label_visibility="collapsed")
    st.divider()
    st.caption(f"Accounts: **{len(scores):,}**  ·  Transactions: **{bundle.metrics['n_transactions']:,}**")
    st.caption(f"AI drafting: **{'available' if API_KEY_PRESENT else 'off (no key)'}**")

if page.startswith("🌐"):
    st.markdown(
        landing.welcome_hero(
            "Bank accounts · Bitcoin · one investigation lab",
            "Follow the Money",
            "Money that is laundered leaves a trail of many small moves. This lab reads that trail in bank "
            "transfers and in Bitcoin, scores it, and explains each decision to a human analyst.",
        ),
        unsafe_allow_html=True,
    )

    if not st.session_state.get("signed_in"):
        st.markdown("#### Sign in")
        with st.form("login_form"):
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            if check_login(username, password):
                st.session_state["signed_in"] = True
                st.session_state["user"] = username.strip()
                st.rerun()
            else:
                st.error("Wrong username or password.")
        st.caption(f"Dummy authentication for this prototype. Demo login: **demo** / **{'demo123'}**. "
                   "Nothing is stored, and this is not real security.")
    else:
        st.success(f"Signed in as {st.session_state.get('user')}.")
        st.button("Sign out", on_click=sign_out)

    st.markdown("#### The story")
    st.markdown(landing.flow_html(landing.BRIDGE_STEPS), unsafe_allow_html=True)
    st.caption("The bank and Bitcoin datasets are separate and not linked to each other. The story is that the "
               "same kind of pattern shows up in both, so one lab can study both.")

    st.markdown("#### What you can check, and how")
    for what, how in landing.HOW_TO:
        st.markdown(f"- **{what}**: {how}")

    st.markdown("#### How the bank side works")
    st.markdown(landing.flow_html(), unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.markdown("#### What it does")
        st.markdown(
            "- Builds a **transaction graph** of who pays whom.\n"
            "- Scores every account with a **risk model**.\n"
            "- Sends risky accounts to **agents** that look at their network and retrieve typology guidance.\n"
            "- Writes a **case narrative** that cites the guidance it used."
        )
    with b:
        st.markdown("#### What it does not do")
        st.markdown(
            "- It does not decide that a crime happened. It ranks cases for human review.\n"
            "- It is trained on synthetic bank data and real Bitcoin data, and is a learning project, not a compliance system.\n"
            "- The writer uses the AI model only when a key is set; otherwise a fixed template is used."
        )

    st.markdown("#### Start here")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.button("👤 My bank accounts", on_click=go_to, args=("👤 My accounts",))
    with s2:
        st.button("👛 My Bitcoin transactions", on_click=go_to, args=("₿ Crypto (Elliptic)",))
    with s3:
        st.button("🛰️ Watch the system run", on_click=go_to, args=("🛰️ System live",))

elif page.startswith("👤"):
    st.markdown(
        landing.welcome_hero(
            "My accounts",
            "Were any of my accounts used for laundering?",
            "Pick an account to see its risk score, whether it needs review, and the agents' reasoning.",
        ),
        unsafe_allow_html=True,
    )
    if not st.session_state.get("signed_in"):
        st.info("Sign in on the welcome page to see your accounts.")
        st.button("Go to sign-in", on_click=go_to, args=("🌐 Welcome",))
    else:
        st.success(f"Signed in as {st.session_state.get('user')}. Accounts linked to this profile:")
        rows = []
        for acc in demo_accounts(labels):
            s = scores[acc]
            if s >= DECISION_THRESHOLD:
                state_txt = "🚩 Flagged for review"
            elif s >= LOW_RISK_CUTOFF:
                state_txt = "👀 Being watched"
            else:
                state_txt = "✅ Normal"
            rows.append({"account": acc, "risk score": round(float(s), 3), "status": state_txt,
                         "known laundering activity (demo only)": "yes" if labels.get(acc, 0) else "no"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("The last column comes from the dataset labels. A real customer view would not show it.")

        chosen = st.selectbox("Open an account to see the agents' reasoning", [r["account"] for r in rows])
        result = investigate(bundle.as_agent_bundle(), chosen)
        g1, g2 = st.columns([1, 2])
        with g1:
            st.plotly_chart(risk_gauge_figure(result["score"]), key="my-gauge")
            badge = "badge-esc" if result["disposition"] == "ESCALATE" else "badge-clr"
            st.markdown(f'<div style="text-align:center"><span class="{badge}">{result["disposition"]}</span></div>',
                        unsafe_allow_html=True)
        with g2:
            st.plotly_chart(workflow_figure(status_for(result), {}), key="my-wf")
            st.markdown(result.get("narrative", ""))

elif page.startswith("🛰"):
    st.markdown(
        landing.welcome_hero(
            "System live",
            "The whole process, running now",
            "A sample of accounts goes through every stage. Nothing is replayed: each stage runs as you watch.",
        ),
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([2, 1, 1])
    n_run = c1.slider("Accounts to process", 20, 200, 60, step=10)
    use_llm = c2.toggle("AI writer", value=False, disabled=not API_KEY_PRESENT, key="sys_llm")
    start = c3.button("▶ Run the process", type="primary")

    left, right = st.columns([1, 2])
    story_slot = left.empty()
    body = right.container()
    story_slot.markdown(landing.story_html(None), unsafe_allow_html=True)

    if start:
        sample = labels.sample(n_run, random_state=7).index.tolist()
        sample_scores = scores.loc[sample]
        sample_labels = labels.loc[sample]
        m = bundle.metrics

        story_slot.markdown(landing.story_html(0), unsafe_allow_html=True)
        with body:
            k = st.columns(3)
            k[0].metric("Transactions", f"{m['n_transactions']:,}")
            k[1].metric("Accounts", f"{m['n_accounts']:,}")
            k[2].metric("Illicit accounts (labels)", f"{int(labels.sum()):,}")
        time.sleep(1.2)

        story_slot.markdown(landing.story_html(1), unsafe_allow_html=True)
        with body:
            st.plotly_chart(workflow_figure({n: "pending" for n in NODE_ORDER}), key="sys-net")
            st.caption(f"Graph: {bundle.graph.number_of_nodes():,} accounts, "
                       f"{bundle.graph.number_of_edges():,} payment links.")
        time.sleep(1.2)

        story_slot.markdown(landing.story_html(2), unsafe_allow_html=True)
        bar = body.progress(0.0, text="Computing features for the sample…")
        for i in range(1, 11):
            bar.progress(i / 10, text=f"Account features: {int(i / 10 * n_run)}/{n_run}")
            time.sleep(0.15)
        bar.empty()
        time.sleep(0.3)

        story_slot.markdown(landing.story_html(3), unsafe_allow_html=True)
        with body:
            st.plotly_chart(score_distribution_figure(sample_scores, sample_labels), key="sys-scores")
            st.caption("Each sampled account now has a risk score between 0 and 1.")
        time.sleep(1.2)

        story_slot.markdown(landing.story_html(4), unsafe_allow_html=True)
        llm = None
        if use_llm:
            import anthropic

            llm = anthropic.Anthropic()
        agent_bundle = bundle.as_agent_bundle()
        diagram = body.empty()
        prog = body.progress(0.0, text="Starting agents…")
        results = []
        for idx, acc in enumerate(sample, start=1):
            prog.progress(idx / len(sample), text=f"Investigating {idx}/{len(sample)}: {acc}")
            status = {n: "pending" for n in NODE_ORDER}
            final: dict = {}
            for node, update in stream_investigation(agent_bundle, acc, llm=llm):
                if node == DONE_KEY:
                    final = update
                    break
                status[node] = "done"
                if node == "scorer" and update["score"] < LOW_RISK_CUTOFF:
                    for s in ("graph_investigator", "retriever", "writer"):
                        status[s] = "skipped"
                diagram.plotly_chart(workflow_figure(status), key=f"sys-{idx}-{node}")
            results.append({
                "account": acc, "score": round(float(final["score"]), 3),
                "decision": final["disposition"], "writer": final.get("writer_mode", "none"),
                "truth": "illicit" if sample_labels[acc] else "clean",
            })
        prog.empty()
        time.sleep(0.5)

        story_slot.markdown(landing.story_html(5), unsafe_allow_html=True)
        res = pd.DataFrame(results)
        tp = int(((res.decision == "ESCALATE") & (res.truth == "illicit")).sum())
        fp = int(((res.decision == "ESCALATE") & (res.truth == "clean")).sum())
        tn = int(((res.decision == "CLEAR") & (res.truth == "clean")).sum())
        fn = int(((res.decision == "CLEAR") & (res.truth == "illicit")).sum())
        with body:
            k = st.columns(4)
            k[0].metric("Escalated", int((res.decision == "ESCALATE").sum()))
            k[1].metric("Cleared", int((res.decision == "CLEAR").sum()))
            k[2].metric("Accuracy", f"{(tp + tn) / len(res):.1%}")
            k[3].metric("Recall on illicit", f"{tp / (tp + fn):.1%}" if tp + fn else "n/a")
            st.plotly_chart(confusion_figure(tp, fp, tn, fn), key="sys-cm")
            st.dataframe(res, hide_index=True)
        story_slot.markdown(landing.story_html(None), unsafe_allow_html=True)
        st.toast("Process complete")

@st.cache_resource(show_spinner="Loading Bitcoin transaction graph…")
def get_crypto():
    from aml_copilot.crypto import load_crypto_bundle
    return load_crypto_bundle()


if page.startswith("₿"):
    from aml_copilot.crypto import artifacts_ready, build_crypto_artifacts, neighbourhood
    from aml_copilot.crypto.elliptic import demo_transactions

    hero("Bitcoin: follow the transactions",
         "Real data. The Elliptic dataset labels Bitcoin transactions as illicit (scams, ransomware, "
         "Ponzi schemes) or licit. Every transaction is scored, and you can inspect its neighbours.",
         ["Elliptic dataset", "per-transaction scoring", "temporal split", "real labels"])
    if not artifacts_ready():
        st.warning("Crypto artifacts not built yet. Data must be in data/raw/elliptic/.")
        if st.button("Build crypto artifacts (about 2 minutes)", type="primary"):
            with st.spinner("Training on labelled transactions in steps 1–34 and scoring everything…"):
                build_crypto_artifacts()
            st.rerun()
    else:
        cb = get_crypto()
        cm = cb.metrics
        tab_over, tab_mine, tab_explore = st.tabs(["📖 Overview", "👛 My transactions", "🔎 Explorer"])

        with tab_over:
            st.markdown("#### The story in Bitcoin")
            st.markdown(landing.flow_html(landing.CRYPTO_FLOW_STEPS), unsafe_allow_html=True)
            st.markdown(
                "Bitcoin records every payment on a public ledger, but not who owns the wallets. Analysts "
                "look at the shape of payments: a transaction that receives from and pays to known-bad "
                "activity is risky. The labels come from real investigations."
            )
            k = st.columns(5)
            k[0].metric("Transactions", f"{cm['n_transactions']:,}")
            k[1].metric("Labelled", f"{cm['n_labelled']:,}")
            k[2].metric("Illicit share (labelled)", f"{cm['illicit_rate_labelled']:.1%}")
            k[3].metric("ROC-AUC (test)", cm["roc_auc"], help=f"Test = time steps {cm['split_step'] + 1}–49")
            k[4].metric("Illicit F1 (test)", cm["f1_at_threshold"])
            left, right = st.columns([3, 2])
            with left:
                st.subheader("How well do scores separate illicit transactions? (test period)")
                test = cb.scores[cb.scores["split"] == "test"]
                st.plotly_chart(score_distribution_figure(cb.risk[test.index], cb.labels01[test.index]),
                                key="crypto-dist")
            with right:
                st.subheader("Reading this page")
                st.markdown(
                    "- Each **node** is one Bitcoin transaction. Colour is its risk score.\n"
                    "- The model trains only on early time steps and is tested on later ones, "
                    "so it cannot see the future.\n"
                    "- **Unknown** transactions are scored but have no label, so they are never used to measure accuracy."
                )

        with tab_mine:
            if not st.session_state.get("signed_in"):
                st.info("Sign in on the welcome page to see your transactions.")
                st.button("Go to sign-in", on_click=go_to, args=("🌐 Welcome",), key="crypto-to-login")
            else:
                st.success(f"Signed in as {st.session_state.get('user')}. Transactions linked to this profile:")
                rows = []
                for t in demo_transactions(cb):
                    s = float(cb.risk[t])
                    state_txt = ("🚩 Flagged for review" if s >= DECISION_THRESHOLD
                                 else "👀 Being watched" if s >= LOW_RISK_CUTOFF else "✅ Normal")
                    rows.append({"txId": t, "risk score": round(s, 3), "status": state_txt,
                                 "time step": int(cb.scores.loc[t, "time_step"]),
                                 "known illicit (demo only)": "yes" if cb.scores.loc[t, "label"] == "illicit" else "no"})
                st.dataframe(pd.DataFrame(rows), hide_index=True)
                st.caption("The last column comes from the dataset labels. A real customer view would not show it.")
                chosen = st.selectbox("Open a transaction", [r["txId"] for r in rows], key="mine-tx")
                info = neighbourhood(cb, chosen)
                m = st.columns(4)
                m[0].metric("Risk", f"{cb.risk[chosen]:.3f}")
                m[1].metric("Time step", int(cb.scores.loc[chosen, "time_step"]))
                m[2].metric("Inbound links", info["in_degree"])
                m[3].metric("Outbound links", info["out_degree"])
                st.plotly_chart(ego_network_figure(cb.graph, cb.risk.rename(str), cb.labels01.rename(str),
                                                   str(chosen), hops=1, max_nodes=40), key="mine-ego")
                t1, t2 = st.columns(2)
                t1.markdown("**Received from (inbound)**")
                t1.dataframe(pd.DataFrame(info["inbound"]), hide_index=True)
                t2.markdown("**Sent to (outbound)**")
                t2.dataframe(pd.DataFrame(info["outbound"]), hide_index=True)

        with tab_explore:
            ranked = cb.risk.sort_values(ascending=False)
            ranked = ranked[cb.scores.loc[ranked.index, "label"] != "unknown"].head(2000).index.tolist()
            c1, c2 = st.columns([3, 1])
            tx = c1.selectbox(
                "Transaction (labelled, highest risk first)", ranked,
                format_func=lambda t: f"{t}  ·  risk {cb.risk[t]:.3f}  ·  {cb.scores.loc[t, 'label']}",
                key="explore-tx",
            )
            hops = c2.slider("Hops", 1, 3, 2, key="crypto-hops")
            st.plotly_chart(ego_network_figure(cb.graph, cb.risk.rename(str), cb.labels01.rename(str),
                                               str(tx), hops=hops, max_nodes=80), key="crypto-ego")
            info = neighbourhood(cb, tx)
            m = st.columns(4)
            m[0].metric("Risk", f"{cb.risk[tx]:.3f}")
            m[1].metric("Time step", int(cb.scores.loc[tx, "time_step"]))
            m[2].metric("Inbound links", info["in_degree"])
            m[3].metric("Outbound links", info["out_degree"])
            t1, t2 = st.columns(2)
            t1.markdown("**Sent from (inbound)**")
            t1.dataframe(pd.DataFrame(info["inbound"]), hide_index=True)
            t2.markdown("**Sent to (outbound)**")
            t2.dataframe(pd.DataFrame(info["outbound"]), hide_index=True)
            st.caption(f"Test metrics: precision {cm['precision_at_threshold']:.1%}, recall "
                       f"{cm['recall_at_threshold']:.1%} at threshold {cm['decision_threshold']}.")

if page.startswith("💬"):
    from aml_copilot.assistant import CaseChat, account_facts, transaction_facts
    from aml_copilot.copilot.agents import _graph_summary
    from aml_copilot.copilot.model import FEATURES as _FEATS
    from aml_copilot.crypto import artifacts_ready, neighbourhood
    from aml_copilot.crypto.elliptic import demo_transactions

    hero("Ask the assistant",
         "Ask about one account or transaction in your profile: why it was flagged, what to do next, "
         "or what the data can and cannot tell you. Answers come from fixed facts, and the case note "
         "keeps the conversation consistent.",
         ["grounded answers", "case notes", "no guessing", "safe guidance"])
    if not st.session_state.get("signed_in"):
        st.info("Sign in on the welcome page to use the assistant.")
    else:
        kind = st.radio("About", ["Bank account", "Bitcoin transaction"], horizontal=True, key="chat_kind")
        if kind == "Bank account":
            subject = st.selectbox("Account", demo_accounts(labels), key="chat_acc")
            key = f"acc:{subject}"
            row = bundle.features.loc[[subject]]
            facts = account_facts(
                account=subject, score=float(scores[subject]),
                flags=bundle.model.elevated_features(row),
                summary=_graph_summary(bundle.graph, bundle.features, subject),
                known_label="illicit" if labels.get(subject, 0) else "clean",
            )
        else:
            if not artifacts_ready():
                st.warning("Bitcoin artifacts are not built yet. Build them on the Bitcoin page first.")
                st.stop()
            cb = get_crypto()
            subject = st.selectbox("Transaction", demo_transactions(cb), key="chat_tx")
            key = f"tx:{subject}"
            info = neighbourhood(cb, subject)
            facts = transaction_facts(
                tx=subject, score=float(cb.risk[subject]),
                time_step=int(cb.scores.loc[subject, "time_step"]),
                in_links=info["in_degree"], out_links=info["out_degree"],
                known_label=cb.scores.loc[subject, "label"],
            )

        use_ai = st.toggle("Use AI for open questions", value=API_KEY_PRESENT,
                           disabled=not API_KEY_PRESENT, key="chat_ai")
        if st.session_state.get("chat_key") != key:
            llm = None
            if use_ai:
                import anthropic
                llm = anthropic.Anthropic()
            st.session_state["chat_key"] = key
            st.session_state["chat"] = CaseChat(facts=facts, kb=bundle.kb, llm=llm)

        chat = st.session_state["chat"]
        chat.llm = None
        if use_ai:
            import anthropic
            chat.llm = anthropic.Anthropic()

        st.caption("Quick questions:")
        qcols = st.columns(4)
        for col, q in zip(qcols, ["Why was this flagged?", "What can I do next?",
                                  "How do I avoid being flagged?", "What is my balance?"]):
            col.button(q, key=f"q-{q}",
                       on_click=lambda q=q: st.session_state.update(chat_pending=q))

        pending = st.session_state.pop("chat_pending", None)
        if pending:
            chat.ask(pending)

        for role, text in chat.turns:
            with st.chat_message("user" if role == "user" else "assistant"):
                st.write(text)
        prompt = st.chat_input(f"Ask about {chat.subject}")
        if prompt:
            chat.ask(prompt)
            st.rerun()

        with st.expander("Case notes kept for this chat", expanded=False):
            st.caption("Facts are fixed when the chat starts. Each question is logged here.")
            for note in chat.notes:
                st.markdown(f"- {note}")
            if st.button("Clear conversation", key="chat_clear"):
                st.session_state.pop("chat", None)
                st.session_state.pop("chat_key", None)
                st.rerun()

if page.startswith("🏠"):
    m = bundle.metrics
    hero(
        "Bank accounts · command center",
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
         ["LangGraph", "conditional routing", "RAG", "AI writer"])
    c1, c2 = st.columns([3, 1])
    account = c1.selectbox(
        "Account", ranked_accounts[:2000],
        format_func=lambda a: f"{a}  ·  risk {scores[a]:.3f}  ·  {'illicit' if labels.get(a, 0) else 'clean'}",
        key="live_account",
    )
    use_llm = c2.toggle("AI writer", value=API_KEY_PRESENT, disabled=not API_KEY_PRESENT)
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
    use_llm = c2.toggle("AI writer", value=False, disabled=not API_KEY_PRESENT, key="play_llm")
    speed = c3.select_slider("Speed", options=[3.0, 2.0, 1.0, 0.6, 0.3], value=3.0,
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
