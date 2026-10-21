"""TF-IDF + logistic regression: a 2-minute CPU sanity check of the whole pipeline and a floor for CV.

Not an approved backbone — use it to debug and to compare against, never as the submitted model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..config import ExperimentConfig


def _inputs(cfg: ExperimentConfig, df: pd.DataFrame) -> list[str]:
    if cfg.prepend_language:
        return (df["language"].astype(str).map(lambda x: f"__lang_{x} ") + df["text"]).tolist()
    return df["text"].tolist()


def train_fold(
    cfg: ExperimentConfig,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    val_df: pd.DataFrame,
    y_val: np.ndarray,
    test_df: pd.DataFrame,
    seed: int,
):
    # char n-grams catch obfuscated / misspelled attack phrases in every language without a tokenizer
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=300_000, sublinear_tf=True
    )
    tr = _inputs(cfg, train_df)
    X_tr = hstack([word.fit_transform(tr), char.fit_transform(tr)]).tocsr()

    def transform(df: pd.DataFrame):
        t = _inputs(cfg, df)
        return hstack([word.transform(t), char.transform(t)]).tocsr()

    class_weight = None if cfg.class_weight == "none" else "balanced"
    clf = LogisticRegression(
        C=cfg.tfidf.get("C", 8.0), max_iter=3000, class_weight=class_weight, random_state=seed
    )
    clf.fit(X_tr, y_train)
    return (
        clf.predict_proba(transform(val_df)),
        clf.predict_proba(transform(test_df)),
        {"n_features": X_tr.shape[1]},
    )
