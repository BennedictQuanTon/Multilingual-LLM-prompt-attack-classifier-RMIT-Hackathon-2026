"""Decision rule tuned for macro F1.

argmax of the probabilities maximises accuracy, not macro F1. With PROMPT_INJECTION at 7% the model is
under-confident on it, so adding a per-class offset to the log-probabilities before argmax usually gains
a few points of macro F1 for free. Offsets are fitted on out-of-fold predictions only, never on the test set.
"""

from __future__ import annotations

import numpy as np

from .metrics import macro_f1

DEFAULT_GRID = np.round(np.arange(-3.0, 3.0001, 0.05), 4)


def predict_with_bias(probs: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    logp = np.log(np.clip(probs, 1e-9, 1.0))
    return np.argmax(logp if bias is None else logp + bias, axis=1)


def fit_bias(
    probs: np.ndarray, y: np.ndarray, grid: np.ndarray = DEFAULT_GRID, rounds: int = 3
) -> tuple[np.ndarray, float]:
    """Coordinate ascent on the offsets of classes 1..C-1 (class 0 is the reference)."""
    bias = np.zeros(probs.shape[1])
    best = macro_f1(y, predict_with_bias(probs, bias))
    for _ in range(rounds):
        improved = False
        for c in range(1, probs.shape[1]):
            for b in grid:
                trial = bias.copy()
                trial[c] = b
                score = macro_f1(y, predict_with_bias(probs, trial))
                if score > best + 1e-9:
                    best, bias, improved = score, trial, True
        if not improved:
            break
    return bias, best


def cross_fit_bias_score(probs: np.ndarray, y: np.ndarray, folds: np.ndarray) -> float:
    """Honest estimate of the tuned rule: fit offsets on the other folds, score on the held-out fold."""
    pred = np.empty(len(y), dtype=int)
    for k in np.unique(folds):
        va = folds == k
        bias, _ = fit_bias(probs[~va], y[~va])
        pred[va] = predict_with_bias(probs[va], bias)
    return macro_f1(y, pred)
