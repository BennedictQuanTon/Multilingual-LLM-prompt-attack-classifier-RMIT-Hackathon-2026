"""Every model exposes the same function:

    train_fold(cfg, train_df, y_train, val_df, y_val, test_df, seed) -> (val_probs, test_probs, info)

Probabilities are float arrays of shape (n_rows, 3) in the order of constants.LABELS.
"""

from __future__ import annotations

from collections.abc import Callable


def get_trainer(kind: str) -> Callable:
    if kind == "tfidf":
        from .tfidf import train_fold
    else:  # torch / transformers are imported only when actually needed
        from .transformer import train_fold
    return train_fold
