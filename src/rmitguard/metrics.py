"""Competition metric (macro F1) and the breakdowns we use to decide what to try next."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .constants import LABELS


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0))


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(LABELS))), zero_division=0
    )
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=list(LABELS))


def per_language_table(y_true: np.ndarray, y_pred: np.ndarray, language: np.ndarray) -> pd.DataFrame:
    rows = []
    for lang in sorted(set(language)):
        m = language == lang
        rows.append(
            dict(
                language=lang,
                n=int(m.sum()),
                macro_f1=macro_f1(y_true[m], y_pred[m]),
                **{
                    f"f1_{lab}": f
                    for lab, f in zip(
                        LABELS,
                        f1_score(
                            y_true[m],
                            y_pred[m],
                            labels=list(range(len(LABELS))),
                            average=None,
                            zero_division=0,
                        ),
                    )
                },
            )
        )
    return pd.DataFrame(rows).set_index("language")


def confusion_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    return pd.DataFrame(cm, index=[f"true_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS])


def bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n: int = 1000, seed: int = 0, sample_size: int | None = None
) -> tuple[float, float]:
    """95% CI of macro F1. Pass sample_size=1800 to see how noisy the 30% public leaderboard is."""
    rng = np.random.default_rng(seed)
    size = sample_size or len(y_true)
    scores = [
        macro_f1(y_true[idx], y_pred[idx]) for idx in (rng.integers(0, len(y_true), size) for _ in range(n))
    ]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))
