"""Build and load the copilot artifacts (features, model, metrics)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import networkx as nx
import pandas as pd

from .agents import AgentBundle
from .data import account_labels, load_transactions
from .graph_features import account_features, build_graph
from .model import ScoringModel, fit_model
from .rag import KnowledgeBase
from .rag_graph import GraphRetriever

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "data" / "raw" / "HI-Small_Trans.csv"
ARTIFACTS = ROOT / "artifacts" / "copilot"


def build_artifacts(raw_csv: Path = RAW_CSV, out_dir: Path = ARTIFACTS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_transactions(raw_csv)
    labels = account_labels(df)
    graph = build_graph(df)
    features = account_features(df, graph)
    model, metrics = fit_model(features, labels)

    features.to_csv(out_dir / "features.csv.gz", compression="gzip")
    labels.to_csv(out_dir / "labels.csv.gz", compression="gzip")
    joblib.dump(model, out_dir / "model.joblib")
    metrics = {**metrics, "source": raw_csv.name, "n_transactions": int(len(df))}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


@dataclass
class Bundle:
    model: ScoringModel
    features: pd.DataFrame
    labels: pd.Series
    graph: nx.DiGraph
    kb: KnowledgeBase
    metrics: dict
    graph_rag: GraphRetriever | None = None

    def as_agent_bundle(self) -> AgentBundle:
        return {
            "model": self.model,
            "features": self.features,
            "graph": self.graph,
            "kb": self.kb,
            "graph_rag": self.graph_rag,
        }


def load_bundle(raw_csv: Path = RAW_CSV, out_dir: Path = ARTIFACTS) -> Bundle:
    model = joblib.load(out_dir / "model.joblib")
    features = pd.read_csv(out_dir / "features.csv.gz", index_col="account", dtype={"account": str})
    labels = pd.read_csv(out_dir / "labels.csv.gz", index_col="account", dtype={"account": str})["label"]
    metrics = json.loads((out_dir / "metrics.json").read_text())
    graph = build_graph(load_transactions(raw_csv))
    kb = KnowledgeBase.from_dir()
    return Bundle(model, features, labels, graph, kb, metrics, graph_rag=GraphRetriever(kb))
