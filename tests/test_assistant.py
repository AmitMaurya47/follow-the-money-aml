"""Offline tests for the grounded assistant: routing, refusals, fixed facts and memory."""
from __future__ import annotations

from aml_copilot.assistant import CaseChat, account_facts, route, rule_answer


def _facts(score=0.93):
    return account_facts(
        account="123:456", score=score,
        flags=[("partner_diversity_out", 4.2), ("scc_size", 3.5)],
        summary={"out_partners": 9, "in_partners": 2, "in_cycle": True, "reciprocity": 0.4, "forward_ratio": 0.3},
        known_label="illicit",
    )


class _Kb:
    def search(self, query, k=2):
        return [{"title": "Fan-out dispersal", "text": "Many small transfers to many partners.", "score": 0.5}][:k]


class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Resp:
    def __init__(self, text):
        self.content = [_Block(text)]


class _FakeLLM:
    def __init__(self, reply="A reply [KB1].", error=None):
        self.calls, self.reply, self.error = [], reply, error
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return _Resp(self.reply)


def test_routing_covers_intents_and_refusal_comes_first():
    assert route("How do I avoid being flagged?") == "avoidance"
    assert route("What is my balance?") == "unavailable"
    assert route("Why was I flagged?") == "reason"
    assert route("What can I do next?") == "steps"
    assert route("Tell me the score") == "score"
    assert route("Where does the data come from?") == "data"
    assert route("hello there") == "general"


def test_whole_word_matching_avoids_substring_false_positives():
    assert route("Is my account alive and well?") == "general"  # not 'live'


def test_unavailable_data_is_said_plainly_and_not_invented():
    chat = CaseChat(facts=_facts())
    out = chat.ask("What is my balance?")
    assert out["intent"] == "unavailable"
    assert "not in our dataset" in out["answer"]
    assert "balance" not in out["answer"].lower().replace("account balance", "")


def test_avoidance_is_declined_with_legitimate_steps():
    out = CaseChat(facts=_facts()).ask("How do I avoid being flagged?")
    assert out["intent"] == "avoidance"
    assert "can't help" in out["answer"]
    assert "records" in out["answer"]


def test_steps_follow_status():
    flagged = rule_answer("steps", "what can i do", _facts(0.93), [])
    normal = rule_answer("steps", "what can i do", _facts(0.02), [])
    assert "Gather documents" in flagged
    assert "No action is needed" in normal


def test_reasons_are_plain_language_from_fixed_flags():
    text = rule_answer("reason", "why", _facts(), [])
    assert "sends money to unusually many different people" in text
    assert "loop of transfers" in text


def test_facts_are_fixed_across_follow_up_questions():
    chat = CaseChat(facts=_facts(0.93), kb=_Kb())
    first = chat.ask("Why was I flagged?")["answer"]
    second = chat.ask("And the score?")["answer"]
    assert "0.93" in first and "0.93" in second
    assert chat.facts["risk_score"] == 0.93
    assert len(chat.turns) == 4 and chat.turns[0] == ("user", "Why was I flagged?")


def test_notes_record_subject_and_each_question():
    chat = CaseChat(facts=_facts())
    chat.ask("What can I do next?")
    assert chat.notes[0].startswith("Bank account 123:456")
    assert any("steps" in n for n in chat.notes[1:])


def test_llm_used_for_open_questions_with_fact_sheet_and_history():
    llm = _FakeLLM(reply="Here is a grounded reply [KB1].")
    chat = CaseChat(facts=_facts(), kb=_Kb(), llm=llm)
    chat.ask("Tell me more about this account")
    chat.ask("And what does that pattern mean?")
    assert llm.calls, "LLM should be called for general questions"
    user_msg = llm.calls[1]["messages"][0]["content"]
    assert "risk_score: 0.93" in user_msg and "RECENT CONVERSATION" in user_msg
    assert "Tell me more" in user_msg  # history carried into the second call


def test_safety_intents_never_reach_the_llm():
    llm = _FakeLLM()
    chat = CaseChat(facts=_facts(), llm=llm)
    chat.ask("How do I avoid being flagged?")
    chat.ask("What is my balance?")
    assert llm.calls == []


def test_llm_failure_falls_back_to_rules_and_says_so():
    chat = CaseChat(facts=_facts(), llm=_FakeLLM(error=RuntimeError("down")))
    out = chat.ask("Tell me about this account")
    assert out["mode"].startswith("rules (llm failed")
    assert "0.93" in out["answer"]
