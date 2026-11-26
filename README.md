# Follow the Money: AML and Bitcoin investigation lab

An end-to-end lab for anti-money-laundering review, built with Python and Streamlit.

- **Bank accounts:** a risk model on the IBM synthetic AML sample (HI-Small), a transaction graph, loop and fan-out detection, and per-account score explanations.
- **Multi-agent investigation:** a LangGraph workflow (scorer, graph investigator, retriever, writer, decision). Every run is logged, and a failed run can be resumed from the agent that crashed.
- **Bitcoin:** a risk model on the real, anonymised Elliptic dataset, with transaction inspection.
- **Case assistant:** answers questions about one account or transaction from a fixed fact sheet, and declines to help avoid review.
- **Responsible use:** outputs are advisory and a human makes the final decision. See [docs/responsible_ai.md](docs/responsible_ai.md).

## Quick start

Full instructions, including dataset download, are in [SETUP.md](SETUP.md). In short:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[ui,dev]"

# Put the datasets in data/raw/ (see SETUP.md, step 4), then:
PYTHONPATH=src python -c "from aml_copilot.copilot import build_artifacts; build_artifacts()"
PYTHONPATH=src python -c "from aml_copilot.crypto import build_crypto_artifacts; build_crypto_artifacts()"

PYTHONPATH=src streamlit run copilot_ui.py
```

Open http://localhost:8501 and sign in with **demo / demo123**. This is dummy sign-in for the prototype only.

An Anthropic API key is optional. Put it in a `.env` file as `ANTHROPIC_API_KEY=...` (copy `.env.example`). Without a key the app uses fixed templates and rules. Never commit `.env`.

## Run the tests

```bash
PYTHONPATH=src python -m pytest tests/test_copilot.py tests/test_runs.py tests/test_assistant.py tests/test_auth.py tests/test_elliptic.py tests/test_ui_pages.py -q
```

Some tests use the built artifacts and are skipped if they are missing.

## Agent run logs and recovery

Each investigation writes a log to `logs/agent_runs/`. From the command line:

```bash
PYTHONPATH=src python scripts/agent_runs.py list
PYTHONPATH=src python scripts/agent_runs.py show <run_id>
PYTHONPATH=src python scripts/agent_runs.py resume <run_id>
```

The Live system & testing page in the app shows the same logs and can inject a failure to demonstrate recovery.

## Project layout

| Path | What it is |
|---|---|
| `copilot_ui.py` | Streamlit app (all pages) |
| `src/aml_copilot/copilot/` | Bank graph, model, retrieval, LangGraph agents, run logs |
| `src/aml_copilot/crypto/` | Bitcoin (Elliptic) scoring |
| `src/aml_copilot/assistant.py` | Case assistant |
| `src/aml_copilot/auth.py` | Dummy sign-in |
| `docs/` | Typology notes, copilot guide, responsible AI, research phases |
| `scripts/` | Run-log tool, Elliptic baseline |
| `tests/` | Offline tests |

## Documentation

- [SETUP.md](SETUP.md): step-by-step setup and troubleshooting
- [docs/responsible_ai.md](docs/responsible_ai.md): guardrails, data limits, audit trail
- [docs/copilot.md](docs/copilot.md): detailed guide to the agents and model
- [docs/research-phases.md](docs/research-phases.md): the earlier research project README
