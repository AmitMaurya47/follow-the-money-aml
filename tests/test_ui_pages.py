"""Offline checks for the landing helpers and the new UI pages (Streamlit AppTest)."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from aml_copilot.copilot import landing
from aml_copilot.copilot.landing import FLOW_STEPS, flow_html, story_html, welcome_hero

ROOT = Path(__file__).resolve().parents[1]


def test_flow_has_every_stage_and_arrows_between_them():
    html = flow_html()
    for _, title, _ in FLOW_STEPS:
        assert title in html
    assert html.count("flow-arrow") == len(FLOW_STEPS) - 1


def test_highlight_marks_exactly_one_step():
    assert flow_html(highlight=2).count("flow-step hl") == 1


def test_flow_escapes_text():
    html = flow_html([("x", "<script>", "a & b")])
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_watermark_is_embedded_svg_and_deterministic():
    assert landing.WATERMARK_URL.startswith("data:image/svg+xml;utf8,")
    assert landing._watermark_svg() == landing._watermark_svg()


def test_story_marks_active_card():
    html = story_html(active=4)
    assert html.count("border-color:#34d399") == 1
    assert story_html(None).count("border-color") == 0


def test_welcome_hero_escapes_title():
    assert "&lt;b&gt;" in welcome_hero("k", "<b>t</b>", "s")


@pytest.mark.skipif(not (ROOT / "artifacts/copilot/model.joblib").exists(), reason="artifacts not built")
def test_pages_render_without_exception():
    at = AppTest.from_file(str(ROOT / "copilot_ui.py"), default_timeout=120).run()
    assert not at.exception
    assert any("How the bank side works" in m.value for m in at.markdown)
    assert any("The story" in m.value for m in at.markdown)

    at.text_input(key="login_user").input("demo")
    at.text_input(key="login_pass").input("demo123")
    at.button[0].click().run()  # dummy login on the welcome page
    assert not at.exception and at.session_state["signed_in"]

    at.sidebar.radio[0].set_value("👤 My accounts").run()
    assert not at.exception
    assert at.selectbox, "signed-in view should offer an account picker"

    at.sidebar.radio[0].set_value("₿ Crypto (Elliptic)").run()
    assert not at.exception

    at.sidebar.radio[0].set_value("🛰️ System live").run()
    assert not at.exception
