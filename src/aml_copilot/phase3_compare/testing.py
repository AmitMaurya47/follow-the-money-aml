"""
Phase 3 offline mini-fixture runner helpers.

FOR TEST USE ONLY.  Do not import from production code.

Provides thin convenience wrappers around the production runners and
the committed mini-fixture paths.  The production runners already support
arbitrary fixture sizes (EXPECTED_EVAL_SIZE is not enforced as a hard
constraint), so these wrappers simply centralise the instantiation pattern.

Mini fixture paths
------------------
    MINI_EVAL_PATH     : tests/fixtures/phase3_mini_eval.jsonl
    MINI_BASELINE_PATH : tests/fixtures/phase3_mini_baseline.jsonl
    MINI_EXPECTED_PATH : tests/fixtures/phase3_expected_comparison.json

These paths are relative to the project root derived from this file's
location (src/aml_copilot/phase3_compare/testing.py → parents[3]).
"""
from __future__ import annotations

from pathlib import Path

from aml_copilot.schemas import Phase3CaseResult

# ── Fixture paths ─────────────────────────────────────────────────────────────
# parents[3] = project root  (src/aml_copilot/phase3_compare/ → src/aml_copilot/ → src/ → root)

_FIXTURES_DIR: Path = Path(__file__).parents[3] / "tests" / "fixtures"

MINI_EVAL_PATH: Path = _FIXTURES_DIR / "phase3_mini_eval.jsonl"
MINI_BASELINE_PATH: Path = _FIXTURES_DIR / "phase3_mini_baseline.jsonl"
MINI_EXPECTED_PATH: Path = _FIXTURES_DIR / "phase3_expected_comparison.json"


# ── Convenience mini runners ──────────────────────────────────────────────────


def run_mini_langgraph(
    eval_path: Path = MINI_EVAL_PATH,
    baseline_path: Path = MINI_BASELINE_PATH,
) -> list[Phase3CaseResult]:
    """Run LangGraph pipeline against a mini fixture set."""
    from aml_copilot.phase3_compare.langgraph_runner import LangGraphRunner
    return LangGraphRunner().run(eval_path, baseline_path)


def run_mini_crewai(
    eval_path: Path = MINI_EVAL_PATH,
    baseline_path: Path = MINI_BASELINE_PATH,
    *,
    verbose: bool = False,
) -> list[Phase3CaseResult]:
    """Run CrewAI pipeline against a mini fixture set."""
    from aml_copilot.phase3_compare.crewai_runner import CrewAIRunner
    return CrewAIRunner(verbose=verbose).run(eval_path, baseline_path)


def run_mini_claude(
    eval_path: Path = MINI_EVAL_PATH,
    baseline_path: Path = MINI_BASELINE_PATH,
    *,
    client=None,
) -> list[Phase3CaseResult]:
    """Run the Claude (claude-haiku-5-5) pipeline against a mini fixture set.

    Requires ANTHROPIC_API_KEY; makes live API calls.
    """
    from aml_copilot.phase3_compare.claude_runner import ClaudeRunner
    return ClaudeRunner(client=client).run(eval_path, baseline_path)
