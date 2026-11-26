"""
Phase 3 — Claude (Anthropic Messages API, tool use) adapter satisfying AMLAgentRunner.

Replaces the OpenAI Agents SDK adapter. Model is fixed to claude-haiku-5-5.

Per EvalCase the adapter runs a short tool-use loop against the Anthropic
Messages endpoint:

  1. apply_aml_policy (tool) — Claude passes the evidence JSON; the runner
     executes mock_llm_call(evidence) locally. This is the only permitted
     source of disposition logic, so the decision policy is identical to the
     LangGraph and CrewAI adapters.
  2. submit_decision (tool) — Claude returns the final structured decision
     (disposition, decision_reason, agent_reasoning, human_review_flagged).

Requires ANTHROPIC_API_KEY in the environment. Unlike the previous offline
adapter, this one makes real HTTP calls, so tokens_used is populated from the
API usage field.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from aml_copilot.phase3_compare._shared import (
    build_evidence,
    load_baseline_results,
    load_eval_cases,
    validate_phase3_results,
)
from aml_copilot.phase3_compare.mock_llm import mock_llm_call
from aml_copilot.schemas import CaseResult, EvalCase, Phase3CaseResult

CLAUDE_MODEL: str = "claude-haiku-5-5"
_FRAMEWORK: str = "claude"
_MAX_TURNS: int = 4
_MAX_TOKENS: int = 1024

_POLICY_TOOL: dict[str, Any] = {
    "name": "apply_aml_policy",
    "description": (
        "Apply the shared AML risk policy to the case evidence. Pass the evidence "
        "JSON exactly as received. Returns the policy output as JSON."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"evidence_json": {"type": "string"}},
        "required": ["evidence_json"],
    },
}

_DECISION_TOOL: dict[str, Any] = {
    "name": "submit_decision",
    "description": "Submit the final AML triage decision for this case.",
    "input_schema": {
        "type": "object",
        "properties": {
            "disposition": {"type": "string", "enum": ["ESCALATE", "CLEAR"]},
            "decision_reason": {"type": "string"},
            "agent_reasoning": {"type": "string"},
            "human_review_flagged": {"type": "boolean"},
        },
        "required": [
            "disposition",
            "decision_reason",
            "agent_reasoning",
            "human_review_flagged",
        ],
    },
}

_SYSTEM_PROMPT: str = (
    "You are an AML triage analyst. For each case: call apply_aml_policy with the "
    "evidence JSON, then call submit_decision using the policy's disposition, "
    "decision_reason, and reasoning. Do not change the policy's disposition."
)


class ClaudeRunner:
    """AMLAgentRunner adapter for the Anthropic Messages API (claude-haiku-5-5)."""

    framework_name: str = _FRAMEWORK

    def __init__(self, *, client: Any | None = None, model: str = CLAUDE_MODEL) -> None:
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            import anthropic  # optional [compare] dependency

            client = anthropic.Anthropic()
        self._client = client
        self._model = model

    def run(self, eval_path: Path, baseline_path: Path) -> list[Phase3CaseResult]:
        if not eval_path.exists():
            raise FileNotFoundError(f"Eval set not found: {eval_path}")
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline results not found: {baseline_path}")

        cases: list[EvalCase] = load_eval_cases(eval_path)
        baseline_map: dict[str, CaseResult] = load_baseline_results(baseline_path)

        results: list[Phase3CaseResult] = []
        for case in cases:
            baseline = baseline_map.get(case.case_id)
            if baseline is None:
                raise RuntimeError(f"Case {case.case_id!r} absent from baseline results")

            evidence_json = json.dumps(build_evidence(baseline))
            t0 = time.perf_counter()
            decision, tokens = self._run_case(case.case_id, evidence_json)
            latency_ms = (time.perf_counter() - t0) * 1000

            results.append(
                Phase3CaseResult(
                    framework=_FRAMEWORK,
                    case_id=case.case_id,
                    account_id=case.account_id,
                    disposition=decision["disposition"],
                    decision_reason=decision["decision_reason"],
                    sanctions_hits=baseline.sanctions_hits,
                    rule_firings=baseline.rule_firings,
                    anomaly_score=baseline.anomaly_score,
                    latency_ms=latency_ms,
                    agent_reasoning=decision["agent_reasoning"],
                    agent_override=decision["disposition"] != baseline.disposition,
                    baseline_disposition=baseline.disposition,
                    human_review_flagged=bool(decision["human_review_flagged"]),
                    tokens_used=tokens,
                    cost_usd=0.0,
                )
            )

        validate_phase3_results(results, cases, _FRAMEWORK, allow_usage=True)
        return results

    def _run_case(self, case_id: str, evidence_json: str) -> tuple[dict[str, Any], int]:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"Case {case_id} evidence:\n{evidence_json}"}
        ]
        policy_output: dict[str, Any] | None = None
        tokens = 0

        for _ in range(_MAX_TURNS):
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=_SYSTEM_PROMPT,
                tools=[_POLICY_TOOL, _DECISION_TOOL],
                messages=messages,
            )
            tokens += _usage_tokens(resp)
            messages.append({"role": "assistant", "content": resp.content})

            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                break

            tool_results: list[dict[str, Any]] = []
            for block in tool_uses:
                if block.name == "submit_decision":
                    decision = dict(block.input)
                    if policy_output is not None:
                        # Policy output is authoritative; Claude only routes the call.
                        decision["disposition"] = policy_output["disposition"]
                        decision["decision_reason"] = policy_output["decision_reason"]
                        decision["agent_reasoning"] = policy_output["reasoning"]
                        decision["human_review_flagged"] = policy_output["human_review"]
                    return decision, tokens
                if block.name == "apply_aml_policy":
                    evidence = json.loads(block.input["evidence_json"])
                    policy_output = dict(mock_llm_call(evidence))
                    content = json.dumps(policy_output)
                else:
                    content = f"Unknown tool {block.name}"
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": content}
                )
            messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(f"Case {case_id!r}: no submit_decision within {_MAX_TURNS} turns")


def _usage_tokens(resp: Any) -> int:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0
    return int(getattr(usage, "input_tokens", 0) or 0) + int(
        getattr(usage, "output_tokens", 0) or 0
    )
