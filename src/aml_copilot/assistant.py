"""Grounded question answering about one account or transaction, with a running case note.

Rules that keep answers from drifting:
- Every answer is built from a fact sheet that is fixed when the chat starts. Facts are never
  re-derived from a new question, so the same subject always gets the same numbers.
- Questions about things the dataset does not hold are answered with a clear "not in the data".
- Questions about avoiding review are declined and redirected to legitimate steps. The same
  applies to questions about identities or legal outcomes.
- An LLM may phrase answers to open questions, but only from the fact sheet, the retrieved
  guidance and the recent conversation. Safety-sensitive intents always use the fixed rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .copilot.agents import LOW_RISK_CUTOFF, WRITER_MODEL
from .copilot.model import DECISION_THRESHOLD

DATASET_NOTE_BANK = (
    "The bank data is the IBM synthetic AML sample (HI-Small). It is simulated, so it "
    "contains no real customers."
)
DATASET_NOTE_BTC = (
    "The Bitcoin data is the Elliptic dataset of real, anonymised transactions with labels from "
    "investigations. It has no wallet owners or personal details."
)

UNAVAILABLE_TOPICS = {
    "personal details": ["my name", "who am i", "address", "phone", "email", "date of birth", "passport"],
    "account balance": ["balance", "how much money do i have", "how much is in"],
    "live or recent activity": ["today", "right now", "yesterday", "this week", "live", "latest transfer"],
    "identity of the other party": ["who is the other", "who received", "who sent", "real name", "identity of"],
    "investigation outcome or legal status": ["am i a criminal", "will i be charged", "legal status", "court", "police"],
}

AVOIDANCE_WORDS = ["avoid", "evade", "get around", "bypass", "not get caught", "not be flagged",
                   "stay under", "escape", "hide my", "without being flagged", "undetected"]
STEP_WORDS = ["what can i do", "what should i do", "steps", "next step", "what now", "how do i fix",
              "how can i", "what do i do"]
REASON_WORDS = ["why", "reason", "flagged", "suspicious", "risky", "risk factor"]
SCORE_WORDS = ["score", "how high", "percent", "number", "rating"]
LINK_WORDS = ["who", "neighbour", "neighbor", "connected", "linked", "partner", "sent", "received"]
DATA_WORDS = ["dataset", "data", "source", "synthetic", "real data", "where does"]

FEATURE_PLAIN = {
    "partner_diversity_out": "sends money to unusually many different people",
    "partner_diversity_in": "receives money from unusually many different people",
    "in_txn_count": "receives an unusually high number of transfers",
    "out_txn_count": "makes an unusually high number of transfers",
    "in_total": "receives a large total amount",
    "out_total": "sends a large total amount",
    "in_out_balance": "money in and money out almost match, as if passing through",
    "reciprocity": "money flows back and forth with the same partners",
    "scc_size": "takes part in a loop of transfers that returns to the same accounts",
    "forward_ratio": "forwards money quickly after receiving it",
    "active_span_days": "has been active over a short, concentrated period",
    "pagerank": "sits at a central position in the network",
    "mean_out_amount": "sends unusually large amounts per transfer",
    "mean_in_amount": "receives unusually large amounts per transfer",
}

STATUS_STEPS = {
    "Flagged for review": [
        "Gather documents for recent large or repeated transfers: invoices, contracts or the purpose of each payment.",
        "Expect an analyst review. Answer questions factually and keep copies of what you provide.",
        "Ask your bank which documents they need and how long the review usually takes.",
    ],
    "Being watched": [
        "No action is required yet. Keep records for unusual or repeated transfers.",
        "If your activity changes, the score may change with it.",
    ],
    "Normal": [
        "No action is needed. Keep normal records of your transfers.",
    ],
}

SYSTEM_PROMPT = (
    "You answer questions for a customer about one account or transaction in an AML review tool. "
    "Use ONLY the FACTS and GUIDANCE provided. If the answer is not in them, say it is not in the data. "
    "Never claim a crime occurred, never guess identities, and never explain how to evade review. "
    "Keep answers under 120 words and cite guidance with its tag, for example [KB1]."
)


def _has(question: str, words: list[str]) -> bool:
    """Whole-word or whole-phrase match, so 'alive' does not match 'live'."""
    return any(re.search(r"\b" + re.escape(w) + r"\b", question) for w in words)


def route(question: str) -> str:
    q = question.lower()
    if _has(q, AVOIDANCE_WORDS):
        return "avoidance"
    if unavailable_topic(q):
        return "unavailable"
    if _has(q, STEP_WORDS):
        return "steps"
    if _has(q, REASON_WORDS):
        return "reason"
    if _has(q, SCORE_WORDS):
        return "score"
    if _has(q, LINK_WORDS):
        return "links"
    if _has(q, DATA_WORDS):
        return "data"
    return "general"


def unavailable_topic(question: str) -> str | None:
    q = question.lower()
    for topic, words in UNAVAILABLE_TOPICS.items():
        if _has(q, words):
            return topic
    return None


def status_for(score: float) -> str:
    if score >= DECISION_THRESHOLD:
        return "Flagged for review"
    if score >= LOW_RISK_CUTOFF:
        return "Being watched"
    return "Normal"


def account_facts(*, account: str, score: float, flags: list[tuple[str, float]],
                  summary: dict[str, Any], known_label: str) -> dict[str, Any]:
    return {
        "kind": "bank account", "id": account, "risk_score": round(float(score), 3),
        "status": status_for(score), "elevated_features": list(flags),
        "out_partners": summary.get("out_partners"), "in_partners": summary.get("in_partners"),
        "in_loop": bool(summary.get("in_cycle")), "reciprocity": summary.get("reciprocity"),
        "forward_ratio": summary.get("forward_ratio"), "known_label": known_label,
        "dataset_note": DATASET_NOTE_BANK,
    }


def transaction_facts(*, tx: int, score: float, time_step: int, in_links: int, out_links: int,
                      known_label: str) -> dict[str, Any]:
    return {
        "kind": "Bitcoin transaction", "id": str(tx), "risk_score": round(float(score), 3),
        "status": status_for(score), "elevated_features": [], "time_step": time_step,
        "in_links": in_links, "out_links": out_links, "known_label": known_label,
        "dataset_note": DATASET_NOTE_BTC,
    }


def _summary_line(f: dict[str, Any]) -> str:
    return (f"{f['kind'].capitalize()} {f['id']}: risk score {f['risk_score']:.2f} "
            f"({f['status'].lower()}).")


def _reasons(f: dict[str, Any]) -> str:
    feats = [FEATURE_PLAIN.get(name, name) for name, _ in f.get("elevated_features", [])]
    if f["kind"] == "bank account":
        if not feats:
            return "No single behaviour stands out strongly against other accounts."
        return "It stands out because it " + "; it ".join(feats) + "."
    return (f"It has {f.get('in_links', 0)} incoming and {f.get('out_links', 0)} outgoing links "
            "to other transactions. The model scores the overall shape of these links.")


def rule_answer(intent: str, question: str, facts: dict[str, Any], guidance: list[dict[str, Any]]) -> str:
    f = facts
    summary = _summary_line(f)
    if intent == "unavailable":
        topic = unavailable_topic(question) or "that detail"
        return (f"That is not in our dataset ({topic}). {f['dataset_note']} "
                f"What we can tell you: {summary} To get {topic}, ask the bank or the service that holds it.")
    if intent == "avoidance":
        return ("I can't help with avoiding a review. What does help: keep records of each transfer's purpose, "
                "understand which pattern triggered the score, and talk to the bank if you think the flag is wrong. "
                f"{summary}")
    if intent == "steps":
        steps = STATUS_STEPS.get(f["status"], STATUS_STEPS["Normal"])
        body = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, start=1))
        return f"{summary}\nWhat you can do:\n{body}"
    if intent == "reason":
        return f"{summary} {_reasons(f)}"
    if intent == "score":
        return (f"{summary} Scores run from 0 to 1. Above {DECISION_THRESHOLD} is flagged for review, "
                f"{LOW_RISK_CUTOFF} to {DECISION_THRESHOLD} is watched, and below {LOW_RISK_CUTOFF} is normal.")
    if intent == "links":
        if f["kind"] == "bank account":
            return (f"{summary} It sends money to {f['out_partners']} partners and receives from "
                    f"{f['in_partners']}. Identities of partners are not in the data.")
        return (f"{summary} It has {f['in_links']} incoming and {f['out_links']} outgoing links. "
                "The other transactions' owners are not in the data.")
    if intent == "data":
        return f"{f['dataset_note']} {summary}"
    tail = f" Relevant guidance: {guidance[0]['title']} [KB1]." if guidance else ""
    return f"{summary} {_reasons(f)}{tail} You can ask about the reasons, the score, the next steps or the links."


def _llm_answer(llm: Any, question: str, facts: dict[str, Any], guidance: list[dict[str, Any]],
                history: list[tuple[str, str]]) -> str:
    fact_lines = "\n".join(f"- {k}: {v}" for k, v in facts.items() if k != "dataset_note")
    guide = "\n".join(f"[KB{i}] {g['title']}: {g['text'][:400]}" for i, g in enumerate(guidance, start=1))
    past = "\n".join(f"{role}: {text}" for role, text in history)
    user = (f"FACTS (fixed for this chat, do not change them):\n{fact_lines}\n"
            f"Dataset note: {facts['dataset_note']}\n\nGUIDANCE:\n{guide or '(none)'}\n\n"
            f"RECENT CONVERSATION:\n{past or '(none)'}\n\nQUESTION: {question}")
    response = llm.messages.create(
        model=WRITER_MODEL, max_tokens=400, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()


@dataclass
class CaseChat:
    """One conversation about one subject. The fact sheet is fixed at construction."""

    facts: dict[str, Any]
    kb: Any | None = None
    llm: Any | None = None
    turns: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.notes.append(_summary_line(self.facts) + f" Dataset: {self.facts['dataset_note']}")

    @property
    def subject(self) -> str:
        return f"{self.facts['kind']} {self.facts['id']}"

    def ask(self, question: str) -> dict[str, Any]:
        q = question.strip()
        if not q:
            raise ValueError("question is empty")
        intent = route(q)
        guidance = self.kb.search(q, k=2) if (self.kb is not None and intent in ("general", "reason")) else []
        mode = "rules"
        if self.llm is not None and intent in ("general", "reason", "links", "score"):
            try:
                answer = _llm_answer(self.llm, q, self.facts, guidance, self.turns[-6:])
                mode = "llm"
            except Exception as exc:  # noqa: BLE001 - keep the chat working offline
                answer = rule_answer(intent, q, self.facts, guidance)
                mode = f"rules (llm failed: {type(exc).__name__})"
        else:
            answer = rule_answer(intent, q, self.facts, guidance)
        self.turns.append(("user", q))
        self.turns.append(("assistant", answer))
        self.notes.append(f"Q ({intent}): {q[:80]}")
        return {"answer": answer, "intent": intent, "mode": mode, "guidance": guidance}
