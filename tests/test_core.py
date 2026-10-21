from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rmitguard.config import ExperimentConfig, load_config
from rmitguard.constants import APPROVED_MODELS, LABELS
from rmitguard.data import load_competition, make_folds, split_fingerprint, text_hash
from rmitguard.metrics import macro_f1
from rmitguard.postprocess import cross_fit_bias_score, fit_bias, predict_with_bias
from rmitguard.submission import make_submission, validate_submission

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_every_config_loads(path):
    cfg = load_config(path)
    assert cfg.model == "tfidf" or cfg.model in APPROVED_MODELS


def test_config_rejects_typos_and_unapproved_models():
    with pytest.raises(ValueError, match="unknown config keys"):
        ExperimentConfig.from_dict({"name": "x", "model": "xlmr-base", "learning_rate": 1e-5})
    with pytest.raises(ValueError, match="not approved"):
        ExperimentConfig(name="x", model="microsoft/deberta-v3-large")
    with pytest.raises(ValueError, match="out of range"):
        ExperimentConfig(name="x", model="xlmr-base", folds=[5])


def test_folds_are_deterministic_grouped_and_stratified(data_dir):
    train = load_competition(data_dir).train
    f1, f2 = make_folds(train, 5, 42), make_folds(train, 5, 42)
    assert (f1 == f2).all() and split_fingerprint(f1) == split_fingerprint(f2)
    assert split_fingerprint(make_folds(train, 5, 7)) != split_fingerprint(f1)
    assert set(f1) == set(range(5))
    hashes = train["text"].map(text_hash)
    assert (pd.DataFrame({"h": hashes, "f": f1}).groupby("h")["f"].nunique() == 1).all()
    for k in range(5):
        assert set(train.loc[f1 == k, "label"]) == set(LABELS)


def test_bias_tuning_recovers_minority_class():
    rng = np.random.default_rng(0)
    y = rng.choice(3, size=3000, p=[0.62, 0.31, 0.07])
    logits = np.eye(3)[y] * 1.5 + rng.normal(0, 1, (3000, 3))
    logits[:, 2] -= 1.5  # under-confident on the rare class, like a model trained on imbalanced data
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    base = macro_f1(y, probs.argmax(1))
    bias, tuned = fit_bias(probs, y)
    assert (predict_with_bias(probs) == probs.argmax(1)).all()
    assert tuned > base + 0.02 and bias[2] > 0
    folds = np.arange(3000) % 5
    assert cross_fit_bias_score(probs, y, folds) > base


def test_submission_validation(data_dir):
    test = pd.read_csv(data_dir / "test.csv")
    good = make_submission(test["id"], np.zeros(len(test), dtype=int))
    assert validate_submission(good, test["id"], expected_rows=len(test)) == []
    bad = good.copy()
    bad.loc[0, "label"] = "benign"
    assert any("invalid labels" in p for p in validate_submission(bad, test["id"], expected_rows=len(test)))
    assert any("rows" in p for p in validate_submission(good.iloc[:-1], test["id"], expected_rows=len(test)))
    assert any("columns" in p for p in validate_submission(good.rename(columns={"label": "pred"})))


def test_head_tail_truncation_keeps_both_ends():
    pytest.importorskip("transformers")
    from rmitguard.models.transformer import head_tail

    ids = list(range(100))
    assert head_tail(ids, 200, 50) == ids
    out = head_tail(ids, 10, 6)
    assert out == [0, 1, 2, 3, 4, 5, 96, 97, 98, 99]
