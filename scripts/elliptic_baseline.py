"""Baseline test on the real Elliptic Bitcoin dataset (no UI, no artifacts).

Labels: class 1 = illicit, 2 = licit, unknown = excluded.
Split: standard temporal split. Train on time steps 1-34, test on 35-49.
Two models: (a) the 166 provided features, (b) same plus in/out degree from the edge list.

Run:
    PYTHONPATH=src .venv/bin/python scripts/elliptic_baseline.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "elliptic"
SPLIT_STEP = 34
SEED = 42


def load() -> tuple[pd.DataFrame, pd.Series]:
    feats = pd.read_csv(DATA / "elliptic_txs_features.csv", header=None)
    feats = feats.rename(columns={0: "txId", 1: "time_step"})
    feats.columns = ["txId", "time_step"] + [f"f{i}" for i in range(feats.shape[1] - 2)]
    classes = pd.read_csv(DATA / "elliptic_txs_classes.csv")
    df = feats.merge(classes, on="txId", how="left")
    df = df[df["class"].isin(["1", "2"])].copy()
    y = (df["class"] == "1").astype(int)
    return df, y


def add_degree_features(df: pd.DataFrame) -> pd.DataFrame:
    edges = pd.read_csv(DATA / "elliptic_txs_edgelist.csv")
    out_deg = edges.groupby("txId1").size().rename("out_degree")
    in_deg = edges.groupby("txId2").size().rename("in_degree")
    df = df.join(out_deg, on="txId").join(in_deg, on="txId")
    return df.fillna({"out_degree": 0, "in_degree": 0})


def evaluate(name: str, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> None:
    train, test = t <= SPLIT_STEP, t > SPLIT_STEP
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                           class_weight="balanced", random_state=SEED)
    model.fit(X[train], y[train])
    proba = model.predict_proba(X[test])[:, 1]
    yt = y[test]
    print(f"\n=== {name} ===")
    print(f"train labelled: {int(train.sum()):,} (illicit {int(y[train].sum()):,})")
    print(f"test labelled:  {int(test.sum()):,} (illicit {int(yt.sum()):,})")
    print(f"ROC-AUC {roc_auc_score(yt, proba):.4f}   PR-AUC {average_precision_score(yt, proba):.4f}")
    print(classification_report(yt, (proba >= 0.5).astype(int),
                                target_names=["licit", "illicit"], digits=3))


def main() -> None:
    df, y = load()
    print(f"labelled transactions: {len(df):,}  illicit rate {y.mean():.2%}")
    t = df["time_step"]
    feature_cols = [c for c in df.columns if c.startswith("f")]
    evaluate("provided features (165)", df[feature_cols], y, t)

    df2 = add_degree_features(df)
    evaluate("features + degree", df2[feature_cols + ["out_degree", "in_degree"]], y, t)


if __name__ == "__main__":
    main()
