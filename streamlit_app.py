"""
AML Investigation Copilot — Streamlit UI.

Run from the repo root:
    streamlit run streamlit_app.py

Pages
-----
Overview        committed Phase 1/2/3 metrics from artifacts/
Eval cases      browse the frozen eval set and the mini fixtures
Live triage     run one case through the shared policy + claude-haiku-5-5
                (requires ANTHROPIC_API_KEY in .env or the sidebar)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MINI_EVAL = ROOT / "tests" / "fixtures" / "phase3_mini_eval.jsonl"
MINI_BASELINE = ROOT / "tests" / "fixtures" / "phase3_mini_baseline.jsonl"
FULL_EVAL = ROOT / "data" / "fixtures" / "eval.jsonl"


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no overwrite)."""
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

st.set_page_config(page_title="AML Investigation Copilot", layout="wide")


# ── Data loaders ─────────────────────────────────────────────────────────────


@st.cache_data
def load_metrics() -> dict | None:
    path = ARTIFACTS / "phase3_comparison_metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_baseline_metrics() -> dict | None:
    path = ARTIFACTS / "metrics_baseline.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── Pages ────────────────────────────────────────────────────────────────────


def page_overview() -> None:
    st.header("Overview")
    metrics = load_metrics()
    baseline = load_baseline_metrics()

    if metrics is None:
        st.warning("artifacts/phase3_comparison_metrics.json not found.")
    else:
        st.subheader("Framework comparison (committed artifact)")
        st.metric("Comparison passed", "YES" if metrics["comparison_passed"] else "NO")
        cols = st.columns(3)
        cols[0].metric("Eval size", metrics["eval_size"])
        cols[1].metric("Dispositions agree", "YES" if metrics["all_dispositions_agree"] else "NO")
        cols[2].metric("Reasoning agrees", "YES" if metrics["all_reasoning_agree"] else "NO")
        rows = [
            {
                "framework": fw["framework"],
                "accuracy": round(fw["disposition_accuracy"], 4),
                "false-clear (weighted)": round(fw["false_clear_rate_weighted"], 4),
                "p50 latency (ms)": round(fw["latency_p50_ms"], 3),
                "p95 latency (ms)": round(fw["latency_p95_ms"], 3),
                "cases": fw["case_count"],
            }
            for fw in metrics["frameworks"]
        ]
        st.dataframe(rows, width="stretch", hide_index=True)

    if baseline is not None:
        st.subheader("Phase 1 baseline (decision table only)")
        st.json(baseline, expanded=False)


def page_eval_cases() -> None:
    st.header("Eval cases")
    source = st.radio(
        "Source",
        ["Frozen eval set (data/fixtures/eval.jsonl)", "Mini fixtures (5 cases)"],
        horizontal=True,
    )
    cases = load_jsonl(FULL_EVAL if source.startswith("Frozen") else MINI_EVAL)
    if not cases:
        st.warning("No cases found for this source.")
        return

    st.caption(f"{len(cases)} cases")
    st.dataframe(
        [
            {
                "case_id": c["case_id"],
                "account_id": c["account_id"],
                "gold_label": c["gold_label"],
                "case_type": c["case_type"],
                "typology": c.get("typology"),
                "conflict_type": c.get("conflict_type"),
            }
            for c in cases
        ],
        width="stretch",
        hide_index=True,
    )
    chosen = st.selectbox("Inspect a case", [c["case_id"] for c in cases])
    st.json(next(c for c in cases if c["case_id"] == chosen))


def page_live_triage() -> None:
    st.header("Live triage (Claude claude-haiku-5-5)")
    st.caption(
        "Runs one case through the shared decision policy with Claude routing the "
        "tool calls. Makes a live API call and bills your Anthropic account."
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        api_key = st.text_input("ANTHROPIC_API_KEY (not saved)", type="password")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.info("Add ANTHROPIC_API_KEY to .env or enter it above.")
        return

    eval_cases = load_jsonl(MINI_EVAL)
    baseline_rows = {r["case_id"]: r for r in load_jsonl(MINI_BASELINE)}
    chosen = st.selectbox("Case", [c["case_id"] for c in eval_cases])

    if st.button("Run triage", type="primary"):
        from aml_copilot.phase3_compare._shared import build_evidence
        from aml_copilot.phase3_compare.claude_runner import ClaudeRunner
        from aml_copilot.phase3_compare.mock_llm import mock_llm_call
        from aml_copilot.schemas import CaseResult

        baseline = CaseResult.model_validate(baseline_rows[chosen])
        evidence = build_evidence(baseline)
        expected = mock_llm_call(evidence)

        with st.spinner("Calling claude-haiku-5-5..."):
            decision, tokens = ClaudeRunner()._run_case(chosen, json.dumps(evidence))

        col1, col2 = st.columns(2)
        col1.metric("Claude disposition", decision["disposition"])
        col2.metric("Policy disposition", expected["disposition"])
        st.write(f"**Decision reason:** {decision['decision_reason']}")
        st.write(f"**Reasoning:** {decision['agent_reasoning']}")
        st.write(f"**Human review flagged:** {decision['human_review_flagged']}")
        st.caption(f"Tokens used: {tokens}")
        with st.expander("Evidence sent to policy"):
            st.json(evidence)


# ── Navigation ───────────────────────────────────────────────────────────────

PAGES = {
    "Overview": page_overview,
    "Eval cases": page_eval_cases,
    "Live triage": page_live_triage,
}

choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.caption("Repo: AML Investigation Copilot")
PAGES[choice]()
