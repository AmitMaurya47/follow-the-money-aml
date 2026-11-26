# Responsible AI use

Written for: analysts and reviewers who use or evaluate the investigation workflow.

## What the system does and does not do

- It ranks accounts and transactions by risk and drafts a case note. It does not block accounts, file reports or decide outcomes.
- Every run ends with `advisory_only` and `human_review_required` set to true. A human analyst makes the final decision.
- A score is a pattern signal, not evidence that a crime occurred. Narratives say so and avoid claiming guilt.

## Guardrails in the code

| Risk | Control |
|---|---|
| The LLM asserts guilt or gives evasion advice | Output guard in `agents.py` (`GUARD_PATTERNS`). A failing draft is replaced by the fixed template and the trace records the replacement. |
| The LLM invents facts | The writer receives only the scored facts and retrieved guidance. The prompt tells it to use only those facts and cite guidance tags. |
| The LLM is unavailable | The workflow falls back to the template and records `writer: llm failed`. The case still completes. |
| Sensitive data in logs | Run logs hold metadata only: node names, timings, scores, writer mode, error types. Narratives, prompts and keys are not logged. |
| Silent failures | Each crash is logged with the agent name and error type. A failed run can be resumed, and the log stays for audit. |
| Assistant answers beyond the data | The case assistant uses a fixed fact sheet. Identity, balance, live activity and legal-outcome questions get a "not in the data" answer. Evasion questions are declined. |

## Data limits

- The bank data is the IBM synthetic AML sample. It has no real customers. Results on it do not show performance on real bank data.
- The Bitcoin data (Elliptic) is real but anonymised. It has no wallet owners.
- Labels come from past investigations and can be wrong or incomplete. Unlabelled items are never used for training or evaluation.
- The model may score some groups differently for reasons unrelated to risk. Check score distributions before using them for any decision about people.

## Audit trail

- `logs/agent_runs/<run_id>.jsonl` records each run: start, each agent, errors, resume, completion, writer mode, guard result, and the review flags.
- `scripts/agent_runs.py show <run_id>` prints a run. Delete the `logs/` folder to remove all run history. It contains account ids, so treat it as internal.

## Before any real use

1. Replace the synthetic data with a governed dataset and document its source and consent basis.
2. Check false positive and false negative rates by segment, and set the threshold with the compliance team.
3. Keep the human review step. Do not connect the scores to automatic blocking or reporting.
4. Review the guard patterns and the fixed templates with the compliance team.
5. Decide the retention period for run logs and delete them on schedule.
