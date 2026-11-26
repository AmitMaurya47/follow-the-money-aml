"""Presentation building blocks for the welcome, sign-in and live-run pages.

Pure strings and small HTML builders, no Streamlit import, so they can be tested directly.
"""
from __future__ import annotations

import random
from html import escape
from urllib.parse import quote

# (icon, title, one-line description) for each stage of the pipeline, in order.
FLOW_STEPS = [
    ("🏦", "Transactions", "Every transfer between accounts"),
    ("🕸️", "Graph", "Who pays whom, how often, and loops"),
    ("📊", "Account features", "16 numbers describing each account"),
    ("🧠", "Risk model", "Gradient boosting gives a score from 0 to 1"),
    ("🚦", "Router", "Low scores are cleared straight away"),
    ("🤖", "Agents", "Graph investigator, RAG retriever, writer"),
    ("🧑‍⚖️", "Analyst decision", "ESCALATE or CLEAR, with a written reason"),
]

# Story shown on the live-run page, one stage per entry.
STORY = [
    ("1. The money moves", "Transfers arrive as a stream: sender, receiver, amount, time."),
    ("2. The network forms", "Each account becomes a point. Each repeated payment becomes a link."),
    ("3. Behaviour is measured", "For every account we count partners, flows, loops and speed of forwarding."),
    ("4. The model scores risk", "A trained model turns those numbers into one risk score per account."),
    ("5. Agents look closer", "Only accounts above the cutoff get investigated, one agent at a time."),
    ("6. A human decides", "The analyst receives a decision, the evidence and a written narrative."),
]


def _watermark_svg(seed: int = 7, nodes: int = 26) -> str:
    rng = random.Random(seed)
    pts = [(rng.uniform(20, 580), rng.uniform(20, 380)) for _ in range(nodes)]
    parts = [
        "<svg xmlns='http://www.w3.org/2000/svg' width='600' height='400' viewBox='0 0 600 400'>",
        "<g stroke='rgb(96,165,250)' stroke-width='1.2' opacity='0.55'>",
    ]
    for _ in range(nodes * 2):
        a, b = rng.sample(range(nodes), 2)
        (x1, y1), (x2, y2) = pts[a], pts[b]
        parts.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}'/>")
    parts.append("</g><g fill='rgb(52,211,153)' opacity='0.8'>")
    for x, y in pts:
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{rng.uniform(2.5, 5.5):.1f}'/>")
    parts.append("</g></svg>")
    return "".join(parts)


WATERMARK_URL = "data:image/svg+xml;utf8," + quote(_watermark_svg())

STYLE = f"""
<style>
.welcome {{
  position: relative; overflow: hidden; padding: 2.2rem 2rem; border-radius: 18px;
  background:
    url("{WATERMARK_URL}") right -40px center / 620px auto no-repeat,
    radial-gradient(1000px 420px at 0% 0%, #1e3a8a 0%, #0f172a 55%, #020617 100%);
  border: 1px solid #1e293b; margin-bottom: 1.2rem;
}}
.welcome h1 {{ margin: 0; font-size: 2.6rem; color: #f8fafc; letter-spacing: -0.5px; }}
.welcome .kicker {{ color: #34d399; font-weight: 600; text-transform: uppercase;
  letter-spacing: 2px; font-size: .78rem; }}
.welcome p {{ color: #cbd5e1; max-width: 640px; font-size: 1.05rem; }}
.flow {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: 6px; margin: 1rem 0; }}
.flow-step {{
  flex: 1 1 120px; min-width: 120px; padding: .8rem .7rem; border-radius: 14px;
  background: #0f172a; border: 1px solid #1e293b; text-align: center;
}}
.flow-step .ico {{ font-size: 1.6rem; }}
.flow-step .t {{ color: #f1f5f9; font-weight: 600; font-size: .9rem; margin-top: 4px; }}
.flow-step .d {{ color: #94a3b8; font-size: .75rem; margin-top: 4px; }}
.flow-step.hl {{ border-color: #34d399; box-shadow: 0 0 0 1px #34d399 inset; }}
.flow-arrow {{ align-self: center; color: #475569; font-size: 1.4rem; }}
.story-card {{
  padding: 1rem 1.1rem; border-radius: 14px; background: #0f172a; border: 1px solid #1e293b;
  margin-bottom: .6rem;
}}
.story-card b {{ color: #f8fafc; }}
.story-card span {{ color: #94a3b8; }}
.signin {{
  padding: 1rem 1.2rem; border-radius: 14px; border: 1px dashed #334155; background: #0b1220;
}}
.disclaimer {{ color: #64748b; font-size: .8rem; }}
</style>
"""


def flow_html(steps: list[tuple[str, str, str]] = FLOW_STEPS, highlight: int | None = None) -> str:
    """Horizontal flowchart of pipeline stages. ``highlight`` marks one stage (0-based)."""
    out = ["<div class='flow'>"]
    for i, (icon, title, desc) in enumerate(steps):
        cls = "flow-step hl" if highlight == i else "flow-step"
        out.append(
            f"<div class='{cls}'><div class='ico'>{icon}</div>"
            f"<div class='t'>{escape(title)}</div><div class='d'>{escape(desc)}</div></div>"
        )
        if i < len(steps) - 1:
            out.append("<div class='flow-arrow'>➜</div>")
    out.append("</div>")
    return "".join(out)


def welcome_hero(kicker: str, title: str, subtitle: str) -> str:
    return (
        f"<div class='welcome'><div class='kicker'>{escape(kicker)}</div>"
        f"<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>"
    )


def story_html(active: int | None = None) -> str:
    """Stacked story cards. ``active`` (0-based) marks the stage currently running."""
    cards = []
    for i, (title, text) in enumerate(STORY):
        border = "border-color:#34d399;" if active == i else ""
        cards.append(
            f"<div class='story-card' style='{border}'><b>{escape(title)}</b><br>"
            f"<span>{escape(text)}</span></div>"
        )
    return "".join(cards)


BRIDGE_STEPS = [
    ("🏦", "Bank accounts", "Money moves in many small steps between accounts"),
    ("🌉", "The bridge", "Laundering often looks the same across money types: fan-out, fan-in, loops"),
    ("₿", "Bitcoin", "The same kind of logic scores each Bitcoin transaction"),
]

CRYPTO_FLOW_STEPS = [
    ("₿", "Transactions", "Each Bitcoin payment is a record on a public ledger"),
    ("🕸️", "Links", "Which payments connect to which others"),
    ("📊", "Features", "166 numbers describing each transaction"),
    ("🧠", "Risk model", "Trained on early months, tested on later ones"),
    ("🧑‍⚖️", "Analyst", "Flagged, watched or normal, with neighbours to inspect"),
]

HOW_TO = [
    ("My bank accounts", "Sign in, then open 👤 My accounts. Pick an account to see its risk status and the agents' reasoning."),
    ("My Bitcoin transactions", "Open ₿ Bitcoin, then 👛 My transactions. Pick one to see its risk and the transactions linked to it."),
    ("Explorer", "In ₿ Bitcoin, 🔎 Explorer lets you search any labelled transaction and inspect its neighbourhood."),
    ("Live system & testing", "🛰️ runs the whole usual process as a demo, or evaluates accuracy on a balanced sample."),
]


# Sidebar pages, in the order a new user should meet them. copilot_ui.py uses this list.
PAGE_NAMES = [
    "🌐 Welcome", "📖 The story", "👤 My accounts", "🛰️ Live system & testing", "🏠 Command center",
    "🕸️ Network", "⚡ Live workflow", "⏯ Playback", "📚 Knowledge base",
    "₿ Crypto (Elliptic)", "💬 Ask the assistant",
]


def story_chapters(bank: dict, btc: dict | None) -> list[dict]:
    """The full story, one chapter per part of the app. Numbers come from the live metrics.

    ``bank`` is the bank model metrics; ``btc`` the Bitcoin metrics, or None if not built yet.
    """
    roc = f"{bank['roc_auc']:.2f}"
    prec = f"{bank['precision_at_threshold']:.0%}"
    rec = f"{bank['recall_at_threshold']:.0%}"
    if btc is not None:
        btc_line = (f"On real Bitcoin transactions the same kind of logic reaches ROC-AUC {btc['roc_auc']:.2f} "
                    f"on transactions from later time steps the model never trained on.")
        btc_test = (f"Precision {btc['precision_at_threshold']:.0%} and recall {btc['recall_at_threshold']:.0%} "
                    f"at threshold {btc['decision_threshold']}, measured on that later period.")
    else:
        btc_line = "Build the Bitcoin artifacts on the Bitcoin page to see its results here."
        btc_test = ""
    return [
        {"title": "Why this exists", "page": "🌐 Welcome",
         "text": ("Criminals hide dirty money by moving it through many small transfers, so each step looks ordinary. "
                  "Banks and crypto exchanges handle millions of transfers, and reviewing them by hand is slow and costly. "
                  "This lab asks one question: can we rank the accounts and transactions most worth a person's time, "
                  "and explain every ranking so the reviewer can check it?")},
        {"title": "Follow the money", "page": "🕸️ Network",
         "text": ("Every transfer links two accounts. Drawn together, the links form a network. Hubs that collect from "
                  "many senders, accounts that fan money out to many receivers, and loops that send money back to where "
                  "it started all stand out in the picture. The Network page lets you explore any account's neighbourhood.")},
        {"title": "Measure each account's behaviour", "page": "🏠 Command center",
         "text": (f"Each of the {bank['n_accounts']:,} bank accounts is described by 16 numbers: how many partners it has, "
                  "how much flows in and out, whether it sits in a loop, how quickly it forwards money. The Command center "
                  "shows the headline numbers and how well the scores separate illicit accounts from clean ones.")},
        {"title": "Turn behaviour into a risk score", "page": "🏠 Command center",
         "text": (f"A gradient-boosted model turns those 16 numbers into a score between 0 and 1. On held-back accounts it "
                  f"reaches ROC-AUC {roc}. Scores are calibrated, so they are meant to read as likelihoods. A high score is "
                  "a reason to look closer. It is not proof of a crime.")},
        {"title": "Explain every score", "page": "👤 My accounts",
         "text": ("A score alone does not help a reviewer. For each account, the 'Why this score?' panel shows which features "
                  "moved it, how far, and in plain language. The reviewer can check the reasoning instead of trusting a number.")},
        {"title": "Agents take the next step", "page": "⚡ Live workflow",
         "text": ("Accounts that score above the cutoff go further. A scorer, a graph investigator, a retriever, a writer and "
                  "a decision step each do one job and pass their findings along. Low-risk accounts skip straight to the "
                  "decision, which keeps the workload focused. The Live workflow page shows each agent as it runs.")},
        {"title": "Replay the reasoning for audit", "page": "⏯ Playback",
         "text": ("Each investigation can be recorded once and replayed step by step. A reviewer or auditor can see exactly "
                  "what each agent saw and produced, in the order it happened.")},
        {"title": "Ground the write-up in evidence", "page": "📚 Knowledge base",
         "text": ("Typology notes cover layering, fan-in and fan-out, circular flows, money mules and reporting guidance. The "
                  "retriever pulls the relevant passages, and the writer cites them, so every claim can be checked against "
                  "a source.")},
        {"title": "Test honestly", "page": "🛰️ Live system & testing",
         "text": (f"The live page runs the whole process on a sample. The Evaluate mode uses a balanced sample, a "
                  f"threshold you can change, and latency numbers, so you can see where the system is accurate and where "
                  f"it misses. At threshold 0.5 the bank model reaches precision {prec} and recall {rec}.")},
        {"title": "The same pattern in Bitcoin", "page": "₿ Crypto (Elliptic)",
         "text": ("Bitcoin records every payment publicly, but not who owns each wallet. The Elliptic dataset adds real "
                  "labels from investigations, so we can test the same idea on real money. " + btc_line)},
        {"title": "Inspect your own transactions", "page": "₿ Crypto (Elliptic)",
         "text": ("After signing in, the My transactions tab lists the Bitcoin transactions linked to your profile, with "
                  "their risk and the neighbouring transactions they connect to. The Explorer tab lets you search any "
                  "labelled transaction. " + btc_test)},
        {"title": "Ask about your case", "page": "💬 Ask the assistant",
         "text": ("Ask why an account was flagged, what you can do next, or what the data cannot tell you. Answers come from "
                  "fixed facts, and the case note keeps the conversation consistent. The assistant will not help anyone "
                  "avoid review. It points to legitimate steps instead.")},
        {"title": "The human decides", "page": "👤 My accounts",
         "text": ("The system ranks, explains and drafts. A person decides. Limits to keep in mind: the bank data is "
                  "synthetic, the Bitcoin data is real but different in shape, the labels come from past investigations "
                  "and may be incomplete, and a score is never proof that a crime happened.")},
    ]
