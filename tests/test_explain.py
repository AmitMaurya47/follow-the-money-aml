"""Offline tests for per-account score explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aml_copilot.copilot.model import FEATURES, explain_score, fit_model


def _fixture():
    rng = np.random.default_rng(3)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES, index=[f"a{i}" for i in range(n)])
    y = (X["out_partners"] > 1.0).astype(int)  # the label depends on one feature only
    X.loc[y == 1, "out_partners"] += 2.0
    model, _ = fit_model(X, y)
    return model, X


def test_explanation_covers_every_feature_and_matches_score():
    model, X = _fixture()
    row = X.iloc[[0]]
    out = explain_score(model, row)
    assert out["score"] == round(model.predict_proba(row), 4)
    assert {d["feature"] for d in out["drivers"]} == set(FEATURES)
    magnitudes = [abs(d["effect"]) for d in out["drivers"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_signal_feature_is_the_top_driver_for_a_flagged_account():
    model, X = _fixture()
    flagged = X[X["out_partners"] > 2.5].iloc[[0]]
    top = explain_score(model, flagged)["drivers"][0]
    assert top["feature"] == "out_partners"
    assert top["effect"] > 0


def test_feature_already_at_median_has_no_effect():
    model, X = _fixture()
    row = X.iloc[[0]].copy()
    row["pagerank"] = model.median["pagerank"]
    effect = next(d for d in explain_score(model, row)["drivers"] if d["feature"] == "pagerank")["effect"]
    assert effect == 0.0
