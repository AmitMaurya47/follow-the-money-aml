"""Run logs and checkpoints for the investigation workflow.

Every investigation gets a run id. While it runs, the store writes:

- ``<run_id>.jsonl``: an append-only event log (run start, each agent start/end, errors,
  resume, completion). Events hold metadata only: node names, timings, scores, writer mode
  and error types. Narratives and prompts are not written to the log.
- ``<run_id>.checkpoint.json``: the case state after the last agent that finished. If an
  agent crashes, ``resume_investigation`` restarts from the agent after that checkpoint, so
  finished work is not repeated.

Logs live under ``logs/agent_runs/`` at the project root and are safe to delete.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = ROOT / "logs" / "agent_runs"


@dataclass
class RunRecord:
    run_id: str
    account_id: str | None
    status: str  # "completed" | "failed" | "interrupted" (log stops with no end marker)
    last_node: str | None  # last agent that finished (checkpoint owner)
    failed_node: str | None
    error: str | None
    checkpoint: dict[str, Any] | None  # case state after ``last_node``


class RunStore:
    def __init__(self, root: Path = RUN_ROOT):
        self.root = Path(root)

    def new_run(self) -> str:
        return uuid.uuid4().hex[:12]

    def _log_path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.jsonl"

    def _checkpoint_path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.checkpoint.json"

    def log(self, run_id: str, event: str, **fields: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                  "run_id": run_id, "event": event, **fields}
        with self._log_path(run_id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def checkpoint(self, run_id: str, node: str, state: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"node": node, "state": state}
        tmp = self._checkpoint_path(run_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        tmp.replace(self._checkpoint_path(run_id))  # atomic: a crash never leaves half a file

    def events(self, run_id: str) -> list[dict[str, Any]]:
        path = self._log_path(run_id)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def load(self, run_id: str) -> RunRecord:
        events = self.events(run_id)
        if not events:
            raise KeyError(f"no run named {run_id!r} in {self.root}")
        account = next((e.get("account_id") for e in events if e["event"] == "run_started"), None)
        last_event = events[-1]["event"]
        # "interrupted" means the log stops without an end marker (for example the process was killed).
        status = {"run_completed": "completed", "run_failed": "failed"}.get(last_event, "interrupted")
        failed = next((e for e in reversed(events) if e["event"] == "node_error"), None)
        cp_path = self._checkpoint_path(run_id)
        cp = json.loads(cp_path.read_text(encoding="utf-8")) if cp_path.exists() else None
        has_failure = failed is not None and status == "failed"
        return RunRecord(
            run_id=run_id,
            account_id=account,
            status=status,
            last_node=cp["node"] if cp else None,
            failed_node=failed["node"] if has_failure else None,
            error=failed["error"] if has_failure else None,
            checkpoint=cp["state"] if cp else None,
        )

    def list_runs(self) -> list[str]:
        if not self.root.exists():
            return []
        logs = sorted(self.root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem for p in logs]  # most recent first


def timer() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
