"""Account-level risk model: gradient boosting over graph features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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

    # Gradient boosting gives sharp scores that pile up near 1 for template-like illicit
    # accounts. Platt (sigmoid) calibration maps them to probabilities with a smoother spread.
    # No class weights: weights would push probabilities away from the real base rate.
    base = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        early_stopping="auto",  # on only for large samples; avoids tiny-fold splits
        random_state=seed,
    )
    # Calibration needs enough examples of each class in every fold; tiny samples skip it.
    folds = min(5, int(y_tr.value_counts().min()))
    if folds >= 2:
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=folds)
        clf.fit(X_tr, y_tr)
        calibration = f"sigmoid (Platt), cv={folds}"
    else:
        clf = base.fit(X_tr, y_tr)
        calibration = "none (too few examples per class)"

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
        "calibration": calibration,
        "illicit_share_scoring_at_0.9": round(float((proba[y_te.values == 1] >= 0.9).mean()), 3),
        "distinct_illicit_scores": int(len(np.unique(proba[y_te.values == 1].round(4)))),
    }

    median = {f: float(X[f].median()) for f in FEATURES}
    q75, q25 = X.quantile(0.75), X.quantile(0.25)
    scale = {f: float(max((q75[f] - q25[f]) / 1.349, 1e-9)) for f in FEATURES}
    return ScoringModel(clf, FEATURES, median, scale), metrics


def explain_score(model: ScoringModel, feature_row: pd.DataFrame) -> dict:
    """Per-account explanation by one-at-a-time replacement.

    For each feature, swap the account's value for the population median and measure how much
    the score changes. A positive effect means that feature pushes the score up. Effects are not
    exactly additive, because features interact, so treat them as "what moves this score".
    """
    base = model.predict_proba(feature_row)
    drivers = []
    for name in model.features:
        swapped = feature_row.copy()
        swapped[name] = model.median[name]
        drivers.append({
            "feature": name,
            "value": round(float(feature_row[name].iloc[0]), 4),
            "population_median": round(float(model.median[name]), 4),
            "effect": round(base - model.predict_proba(swapped), 4),
        })
    drivers.sort(key=lambda d: -abs(d["effect"]))
    return {"score": round(base, 4), "drivers": drivers}
