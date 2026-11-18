"""Account-level risk model: gradient boosting over graph features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "out_txn_count",
    "in_txn_count",
    "out_total",
    "in_total",
    "out_partners",
    "in_partners",
    "mean_out_amount",
    "mean_in_amount",
    "partner_diversity_out",
    "partner_diversity_in",
    "in_out_balance",
    "reciprocity",
    "scc_size",
    "pagerank",
    "forward_ratio",
    "active_span_days",
]
DECISION_THRESHOLD = 0.5
SEED = 42


@dataclass
class ScoringModel:
    estimator: HistGradientBoostingClassifier
    features: list[str]
    # Population median and robust scale per feature, used to explain a score.
    median: dict[str, float]
    scale: dict[str, float]

    def predict_proba(self, feature_row: pd.DataFrame) -> float:
        return float(self.estimator.predict_proba(feature_row[self.features])[:, 1][0])

    def elevated_features(self, feature_row: pd.DataFrame, z_min: float = 3.0) -> list[tuple[str, float]]:
        """Features whose robust z-score is high versus the population, largest first."""
        out = []
        for name in self.features:
            value = float(feature_row[name].iloc[0])
            z = (value - self.median[name]) / self.scale[name]
            if z >= z_min:
                out.append((name, round(z, 2)))
        return sorted(out, key=lambda x: -x[1])


def fit_model(features: pd.DataFrame, labels: pd.Series, seed: int = SEED) -> tuple[ScoringModel, dict]:
    X = features[FEATURES]
    y = labels.reindex(X.index).fillna(0).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
    )
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    k = max(1, int(round(0.01 * len(proba))))
    top_k = np.argsort(-proba)[:k]
    metrics = {
        "n_accounts": int(len(y)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "illicit_rate": round(float(y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_te, proba)), 4),
        "precision_at_threshold": round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "recall_at_threshold": round(float(recall_score(y_te, pred, zero_division=0)), 4),
        "precision_at_top_1pct": round(float(y_te.iloc[top_k].mean()), 4),
        "decision_threshold": DECISION_THRESHOLD,
    }

    median = {f: float(X[f].median()) for f in FEATURES}
    q75, q25 = X.quantile(0.75), X.quantile(0.25)
    scale = {f: float(max((q75[f] - q25[f]) / 1.349, 1e-9)) for f in FEATURES}
    return ScoringModel(clf, FEATURES, median, scale), metrics
