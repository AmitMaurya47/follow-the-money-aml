"""List, inspect and resume agent investigation runs.

Usage (from the project root):

    PYTHONPATH=src python scripts/agent_runs.py list
    PYTHONPATH=src python scripts/agent_runs.py show <run_id>
    PYTHONPATH=src python scripts/agent_runs.py resume <run_id>

``resume`` needs the bank artifacts (see SETUP.md). Add --llm to draft with Claude again,
which needs ANTHROPIC_API_KEY in .env.
"""
from __future__ import annotations

import argparse
import json
import sys

from aml_copilot.copilot.agents import DONE_KEY, resume_investigation
from aml_copilot.copilot.runs import RunStore


def cmd_list(store: RunStore) -> int:
    runs = store.list_runs()
    if not runs:
        print("No runs logged yet.")
    for run_id in runs:
        rec = store.load(run_id)
        where = f" failed at {rec.failed_node}" if rec.status == "failed" else ""
        print(f"{run_id}  {rec.status:<12} account={rec.account_id}{where}")
    return 0


def cmd_show(store: RunStore, run_id: str) -> int:
    rec = store.load(run_id)
    print(f"run {rec.run_id}: {rec.status}, account {rec.account_id}")
    print(f"last finished agent: {rec.last_node or '(none)'}")
    if rec.error:
        print(f"error in {rec.failed_node}: {rec.error}")
    for event in store.events(run_id):
        detail = {k: v for k, v in event.items() if k not in ("ts", "run_id", "event")}
        print(f"{event['ts']}  {event['event']:<14} {json.dumps(detail, default=str)}")
    return 0


def cmd_resume(store: RunStore, run_id: str, use_llm: bool) -> int:
    from aml_copilot.copilot.pipeline import load_bundle

    bundle = load_bundle()
    llm = None
    if use_llm:
        import anthropic

        llm = anthropic.Anthropic()
    final = {}
    for node, update in resume_investigation(bundle.as_agent_bundle(), run_id, llm=llm, store=store):
        if node == DONE_KEY:
            final = update
        else:
            print(f"finished {node}")
    if not final:
        print("resume produced no final state")
        return 1
    print(f"done: {final['disposition']} (score {final['score']:.3f}, writer {final.get('writer_mode')})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    show = sub.add_parser("show")
    show.add_argument("run_id")
    resume = sub.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("--llm", action="store_true", help="draft the narrative with Claude")
    args = parser.parse_args(argv)

    store = RunStore()
    try:
        if args.cmd == "list":
            return cmd_list(store)
        if args.cmd == "show":
            return cmd_show(store, args.run_id)
        return cmd_resume(store, args.run_id, args.llm)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
