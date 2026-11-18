# AML Investigation Copilot — graph, ML, RAG and multi-agent layer

This layer sits alongside the Phase 1–3 triage pipeline. It adds a transaction graph, a risk model, a retrieval knowledge base, a LangGraph multi-agent workflow, and a Streamlit UI.

## Components

| Component | Module | What it does |
|---|---|---|
| Data loader | `copilot/data.py` | Reads IBM AML CSVs (HI-Small, LI-Small, Kaggle export). Account id is `bank:account`. |
| Graph features | `copilot/graph_features.py` | Directed transfer graph (NetworkX). Per-account fan-in/out, pass-through balance, reciprocity, cycle membership (SCC), PageRank, rapid forwarding within 24h. |
| Risk model | `copilot/model.py` | scikit-learn gradient boosting on the features. Labels are used only for training and evaluation. |
| Knowledge base | `copilot/rag.py`, `docs/kb/*.md` | Typology notes split by section. TF-IDF retrieval, runs offline. |
| Agents | `copilot/agents.py` | LangGraph workflow: scorer → (graph investigator → retriever → writer) → decide. Low-risk accounts skip straight to decide. |
| Artifacts | `copilot/pipeline.py` | Builds and loads `artifacts/copilot/` (features, model, metrics). |
| UI | `copilot_ui.py` | Overview, risk leaderboard, account investigation, knowledge base search. |

## Labels and leakage

- An account is illicit if it takes part in any laundering transfer, as sender or receiver. The source labels transactions, not accounts.
- Graph features are unsupervised and computed on the full graph. They never read labels. The model is evaluated on a stratified 30% holdout of accounts.

## Routing and decisions

- Score below 0.15: cleared without investigation.
- Score at or above 0.5: escalated.
- Between those: cleared, with the full investigation trace recorded.

## Writer modes

- `claude`: the writer agent calls `claude-sonnet-5-5` with only the case facts and retrieved passages. Requires `ANTHROPIC_API_KEY`.
- `template`: deterministic narrative. Used when no key is set, or when the Claude call fails (the failure is recorded in the trace).
- `none`: the case was cleared early and no narrative was written.

## Run

```bash
cd AML
PYTHONPATH=src .venv/bin/python -c "from aml_copilot.copilot import build_artifacts; print(build_artifacts())"
PYTHONPATH=src .venv/bin/streamlit run copilot_ui.py
PYTHONPATH=src .venv/bin/python -m pytest tests/test_copilot.py -q
```

## Limitations

- The local file is a ~120K-transaction sample with about 20K accounts and a 9% illicit account rate. The full IBM Kaggle data is much larger and far more imbalanced (about 0.1% laundering), so precision will be lower there. Point `RAW_CSV` in `copilot/pipeline.py` at the full file, or pass a path to `build_artifacts`, and rebuild.
- Retrieval is lexical (TF-IDF). Swapping in an embedding model would be a contained change in `copilot/rag.py`.
- The knowledge base is a short hand-written summary of general typologies. It is not legal or regulatory advice.
