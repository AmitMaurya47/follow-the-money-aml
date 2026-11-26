"""Offline checks that the story covers every page and uses live numbers."""
from __future__ import annotations

from aml_copilot.copilot.landing import PAGE_NAMES, story_chapters

BANK = {"roc_auc": 0.906, "precision_at_threshold": 0.977, "recall_at_threshold": 0.7, "n_accounts": 19980}
BTC = {"roc_auc": 0.9385, "precision_at_threshold": 0.797, "recall_at_threshold": 0.736,
       "f1_at_threshold": 0.765, "decision_threshold": 0.5, "split_step": 34}


def test_every_app_page_has_a_chapter_except_the_story_itself():
    covered = {c["page"] for c in story_chapters(BANK, BTC)}
    assert covered == set(PAGE_NAMES) - {"📖 The story"}


def test_numbers_come_from_metrics():
    text = " ".join(c["text"] for c in story_chapters(BANK, BTC))
    assert "19,980" in text and "0.91" in text and "0.9385" not in text
    assert "0.94" in text and "precision 80%" in text.lower()


def test_bitcoin_chapters_explain_missing_artifacts():
    text = " ".join(c["text"] for c in story_chapters(BANK, None) if "Bitcoin" in c["title"] or "Bitcoin" in c["text"])
    assert "Build the Bitcoin artifacts" in text


def test_story_states_motive_and_limits():
    first = story_chapters(BANK, BTC)[0]["text"]
    last = story_chapters(BANK, BTC)[-1]["text"]
    assert "hide dirty money" in first
    assert "not proof" in " ".join(c["text"] for c in story_chapters(BANK, BTC)) or "never proof" in last
    assert "synthetic" in last
