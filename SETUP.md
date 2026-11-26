# Setup guide: Follow the Money (AML + Bitcoin investigation lab)

This guide gets the project running on a new machine. It takes about 15 minutes, plus the time to download the datasets.

## What you get

- **Bank side:** risk scoring of bank accounts (IBM synthetic AML sample), a transaction graph, LangGraph agents, and a case assistant.
- **Bitcoin side:** risk scoring of real Bitcoin transactions (Elliptic dataset), with neighbour inspection.
- **Streamlit UI:** a welcome page with dummy sign-in, account and transaction pages, a live demo and evaluation page, and an "Ask the assistant" chat.

## 1. Prerequisites

- Python 3.11 or newer
- About 3 GB of free disk space (datasets plus the virtual environment)
- Optional: an Anthropic API key, to let the AI write narratives and answer open questions. Without a key, everything still works using fixed templates and rules.

## 2. Create the environment

From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[ui,dev]"
```

This installs the package, Streamlit, and the test tools.

## 3. Add the API key (optional)

```bash
cp .env.example .env
```

Then open `.env` and add a line:

```
ANTHROPIC_API_KEY=your-key-here
```

Never commit `.env` or share it. The key is read only from this file and never displayed in the app.

## 4. Download the datasets

The datasets are not included in the zip. Place them as shown.

**Bank data (IBM synthetic AML, HI-Small):**
1. Download the "IBM AML" dataset from Kaggle (search "IBM Transactions for Anti Money Laundering").
2. Copy `HI-Small_Trans.csv` into `data/raw/`.

**Bitcoin data (Elliptic, about 690 MB):**
1. Download the three files from the Hugging Face dataset `asjoie/elliptic-bitcoin-dataset`:
   - `elliptic_txs_features.csv`
   - `elliptic_txs_classes.csv`
   - `elliptic_txs_edgelist.csv`
2. Copy all three into `data/raw/elliptic/`.

The final layout should include:

```
data/raw/HI-Small_Trans.csv
data/raw/elliptic/elliptic_txs_features.csv
data/raw/elliptic/elliptic_txs_classes.csv
data/raw/elliptic/elliptic_txs_edgelist.csv
```

## 5. Build the scoring artifacts

Bank artifacts (about 1 minute):

```bash
PYTHONPATH=src python -c "from aml_copilot.copilot import build_artifacts; print(build_artifacts())"
```

Bitcoin artifacts (about 2 minutes): either run the command below, or open the **Bitcoin** page in the app and click "Build crypto artifacts".

```bash
PYTHONPATH=src python -c "from aml_copilot.crypto import build_crypto_artifacts; print(build_crypto_artifacts())"
```

## 6. Run the app

```bash
PYTHONPATH=src streamlit run copilot_ui.py
```

Open the address Streamlit prints (usually http://localhost:8501).

**Demo sign-in:** username `demo`, password `demo123`. This is dummy authentication for the prototype only. It is not real security.

## 7. Run the tests

```bash
PYTHONPATH=src python -m pytest tests/test_copilot.py tests/test_graph_rag.py tests/test_replay.py tests/test_assistant.py tests/test_auth.py tests/test_elliptic.py tests/test_ui_pages.py -q
```

The UI test needs the bank artifacts from step 5. It is skipped if they are missing.

## Troubleshooting

- **"Crypto artifacts not built yet":** run step 5 (Bitcoin) or click the build button on the Bitcoin page.
- **"No such file" for a CSV:** check the file names and folders in step 4. They are case-sensitive on Linux and macOS.
- **Port already in use:** start Streamlit on another port, for example `streamlit run copilot_ui.py --server.port 8502`.
- **Writer says "llm failed" or the AI toggle is off:** the key is missing or invalid. The app falls back to templates, so this is not an error.
- **Out of memory while building the Bitcoin artifacts:** close other programs. The Elliptic feature file is large.

## Project layout

| Path | What it is |
|---|---|
| `copilot_ui.py` | Streamlit app (all pages) |
| `src/aml_copilot/copilot/` | Bank graph, model, RAG, LangGraph agents, story and landing text |
| `src/aml_copilot/crypto/` | Bitcoin (Elliptic) scoring |
| `src/aml_copilot/assistant.py` | Case assistant with fixed facts and memory |
| `src/aml_copilot/auth.py` | Dummy sign-in |
| `docs/` | Typology notes and the detailed copilot guide |
| `scripts/elliptic_baseline.py` | Standalone Bitcoin baseline run |
| `tests/` | Offline tests |

## Run logs and resume

Every investigation writes a log to `logs/agent_runs/`. If an agent crashes, the run can continue from the agent after the last one that finished.

```bash
PYTHONPATH=src python scripts/agent_runs.py list              # all runs and their status
PYTHONPATH=src python scripts/agent_runs.py show <run_id>     # events for one run
PYTHONPATH=src python scripts/agent_runs.py resume <run_id>   # continue a failed run
```

The responsible-AI rules (advisory output, output guard, what is logged) are in `docs/responsible_ai.md`.
