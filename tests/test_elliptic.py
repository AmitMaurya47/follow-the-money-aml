"""Offline tests for the Elliptic transaction scorer, using a tiny synthetic fixture in the same format."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aml_copilot.crypto import build_crypto_artifacts, load_crypto_bundle, neighbourhood


def _write_fixture(root):
    rng = np.random.default_rng(0)
    n = 300
    ids = np.arange(1000, 1000 + n)
    steps = rng.integers(1, 50, size=n)
    labels = rng.choice(["1", "2", "unknown"], size=n, p=[0.1, 0.5, 0.4])
    signal = (labels == "1") * 1.5
    feats = rng.normal(size=(n, 5)) + signal[:, None]
    raw = pd.DataFrame(np.column_stack([ids, steps, feats]))
    raw.to_csv(root / "elliptic_txs_features.csv", header=False, index=False)
    pd.DataFrame({"txId": ids, "class": labels}).to_csv(root / "elliptic_txs_classes.csv", index=False)
    edges = pd.DataFrame({"txId1": ids[:-1], "txId2": ids[1:]})
    edges.to_csv(root / "elliptic_txs_edgelist.csv", index=False)
    return ids, labels, steps


def test_build_and_load_round_trip(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    ids, labels, steps = _write_fixture(data)
    metrics = build_crypto_artifacts(data_dir=data, out_dir=out)
    assert metrics["n_transactions"] == len(ids)
    assert metrics["n_labelled"] == int((labels != "unknown").sum())

    bundle = load_crypto_bundle(data_dir=data, out_dir=out)
    assert len(bundle.scores) == len(ids)
    assert set(bundle.scores["split"]) <= {"train", "test", "unlabelled"}
    # Unknown transactions are scored but never counted as train or test
    unknown = bundle.scores[bundle.scores["label"] == "unknown"]
    assert set(unknown["split"]) == {"unlabelled"}
    # Temporal split: every test transaction comes after every training transaction's step
    test_steps = bundle.scores[bundle.scores["split"] == "test"]["time_step"]
    assert (test_steps > metrics["split_step"]).all()
    assert ((bundle.risk >= 0) & (bundle.risk <= 1)).all()


def test_neighbourhood_lists_chain_neighbours(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    ids, _, _ = _write_fixture(data)
    build_crypto_artifacts(data_dir=data, out_dir=out)
    bundle = load_crypto_bundle(data_dir=data, out_dir=out)
    middle = int(ids[5])
    info = neighbourhood(bundle, middle)
    assert info["in_degree"] == 1 and info["out_degree"] == 1
    assert info["inbound"][0]["txId"] == int(ids[4])
    assert info["outbound"][0]["txId"] == int(ids[6])
    missing = neighbourhood(bundle, -1)
    assert missing["inbound"] == [] and missing["outbound"] == []


def test_demo_transactions_are_test_period_and_mixed(tmp_path):
    from aml_copilot.crypto.elliptic import demo_transactions
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    _write_fixture(data)
    build_crypto_artifacts(data_dir=data, out_dir=out)
    bundle = load_crypto_bundle(data_dir=data, out_dir=out)
    picks = demo_transactions(bundle, n_illicit=2, n_licit=2)
    assert len(picks) == 4 and len(set(picks)) == 4
    assert set(bundle.scores.loc[picks, "split"]) == {"test"}
    assert set(bundle.scores.loc[picks, "label"]) == {"illicit", "licit"}
