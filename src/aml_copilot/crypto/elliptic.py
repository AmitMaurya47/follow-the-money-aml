"""Transaction-level risk scoring on the Elliptic Bitcoin dataset.

Unlike the bank copilot (which scores accounts), this scores each Bitcoin transaction.
Labels: class "1" = illicit, "2" = licit, "unknown" is scored but never used for training
or evaluation.

Split: temporal. Train on time steps 1-34, test on 35-49, matching the standard protocol.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "raw" / "elliptic"
ARTIFACTS = ROOT / "artifacts" / "crypto"
SPLIT_STEP = 34
DECISION_THRESHOLD = 0.5
SEED = 42

FEATURE_FILE = "elliptic_txs_features.csv"
CLASS_FILE = "elliptic_txs_classes.csv"
EDGE_FILE = "elliptic_txs_edgelist.csv"


@dataclass
class CryptoBundle:
    scores: pd.DataFrame  # index txId; columns: time_step, label, risk_score, split
    graph: nx.DiGraph
    metrics: dict

    @property
    def risk(self) -> pd.Series:
        return self.scores["risk_score"]

    @property
    def labels01(self) -> pd.Series:
        """1 = illicit, 0 = licit or unknown. Used for colouring and summaries only."""
        return (self.scores["label"] == "illicit").astype(int)


def _load_features(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    raw = pd.read_csv(data_dir / FEATURE_FILE, header=None)
    raw = raw.rename(columns={0: "txId", 1: "time_step"})
    feature_cols = [f"f{i}" for i in range(raw.shape[1] - 2)]
    raw.columns = ["txId", "time_step", *feature_cols]
    raw = raw.set_index("txId")
    classes = pd.read_csv(data_dir / CLASS_FILE, dtype={"txId": "int64"}).set_index("txId")["class"]
    label = classes.reindex(raw.index).fillna("unknown").map(
        {"1": "illicit", "2": "licit"}).fillna("unknown")
    return raw[feature_cols], raw["time_step"], label


def build_crypto_artifacts(data_dir: Path = DATA_DIR, out_dir: Path = ARTIFACTS) -> dict:
    """Train on labelled transactions in steps 1-34, score every transaction, save artifacts."""
    X, time_step, label = _load_features(data_dir)
    known = label != "unknown"
    split = pd.Series("unlabelled", index=X.index)
    split[known & (time_step <= SPLIT_STEP)] = "train"
    split[known & (time_step > SPLIT_STEP)] = "test"

    train_mask = split == "train"
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED)
    y_train = (label[train_mask] == "illicit").astype(int)
    model.fit(X[train_mask], y_train)

    risk = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    scores = pd.DataFrame({
        "time_step": time_step, "label": label, "risk_score": risk.round(6), "split": split,
    })

    test_mask = split == "test"
    y_test = (label[test_mask] == "illicit").astype(int).to_numpy()
    p_test = risk[test_mask].to_numpy()
    pred = (p_test >= DECISION_THRESHOLD).astype(int)
    metrics = {
        "n_transactions": int(len(X)),
        "n_labelled": int(known.sum()),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "illicit_rate_labelled": round(float((label[known] == "illicit").mean()), 4),
        "test_illicit": int(y_test.sum()),
        "roc_auc": round(float(roc_auc_score(y_test, p_test)), 4),
        "pr_auc": round(float(average_precision_score(y_test, p_test)), 4),
        "precision_at_threshold": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall_at_threshold": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1_at_threshold": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "decision_threshold": DECISION_THRESHOLD,
        "split_step": SPLIT_STEP,
        "source": FEATURE_FILE,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "scores.csv.gz", compression="gzip")
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_crypto_bundle(data_dir: Path = DATA_DIR, out_dir: Path = ARTIFACTS) -> CryptoBundle:
    scores = pd.read_csv(out_dir / "scores.csv.gz", index_col=0, dtype={"label": str, "split": str})
    scores.index = scores.index.astype("int64")
    metrics = json.loads((out_dir / "metrics.json").read_text())
    edges = pd.read_csv(data_dir / EDGE_FILE)
    graph = nx.from_pandas_edgelist(edges, "txId1", "txId2", create_using=nx.DiGraph)
    # String node ids: the shared plotting helpers expect string names.
    graph = nx.relabel_nodes(graph, str)
    return CryptoBundle(scores=scores, graph=graph, metrics=metrics)


def artifacts_ready(out_dir: Path = ARTIFACTS) -> bool:
    return all((out_dir / f).exists() for f in ("scores.csv.gz", "model.joblib", "metrics.json"))


def neighbourhood(bundle: CryptoBundle, tx: int, top_n: int = 8) -> dict:
    """Direct inbound and outbound transactions of ``tx`` with their risk scores."""
    node = str(tx)
    present = node in bundle.graph

    def rows(pairs):
        out = []
        for n in pairs:
            txid = int(n)
            risk = float(bundle.risk.get(txid, np.nan))
            out.append({"txId": txid, "risk": round(risk, 3),
                        "label": bundle.scores["label"].get(txid, "unknown")})
        return sorted(out, key=lambda r: -(r["risk"] if r["risk"] == r["risk"] else -1))[:top_n]

    return {
        "in_degree": bundle.graph.in_degree(node) if present else 0,
        "out_degree": bundle.graph.out_degree(node) if present else 0,
        "inbound": rows(bundle.graph.predecessors(node)) if present else [],
        "outbound": rows(bundle.graph.successors(node)) if present else [],
    }


def demo_transactions(bundle: CryptoBundle, seed: int = 11, n_illicit: int = 3, n_licit: int = 2) -> list[int]:
    """Fixed demo profile drawn from the test period: illicit and licit transactions."""
    test = bundle.scores[bundle.scores["split"] == "test"]
    illicit = test[test["label"] == "illicit"].index.to_series().sample(n_illicit, random_state=seed)
    licit = test[test["label"] == "licit"].index.to_series().sample(n_licit, random_state=seed)
    return [int(t) for t in pd.concat([illicit, licit]).tolist()]
