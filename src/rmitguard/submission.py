"""Build and validate submission files. A malformed file burns one of only 6 daily submissions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import LABELS, N_TEST_ROWS


def make_submission(test_ids: pd.Series | np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"id": np.asarray(test_ids), "label": np.asarray(LABELS)[pred]})


def validate_submission(
    sub: pd.DataFrame | str | Path,
    test_ids: pd.Series | np.ndarray | None = None,
    expected_rows: int | None = N_TEST_ROWS,
) -> list[str]:
    """Return a list of problems; empty list = safe to submit."""
    if not isinstance(sub, pd.DataFrame):
        sub = pd.read_csv(sub)
    problems = []
    if list(sub.columns) != ["id", "label"]:
        problems.append(f"columns must be exactly ['id', 'label'], got {list(sub.columns)}")
        return problems
    if expected_rows is not None and len(sub) != expected_rows:
        problems.append(f"expected {expected_rows} rows, got {len(sub)}")
    if sub["id"].duplicated().any():
        problems.append(f"{int(sub['id'].duplicated().sum())} duplicated ids")
    bad = sorted(set(sub["label"].astype(str)) - set(LABELS))
    if bad:
        problems.append(f"invalid labels {bad}")
    if test_ids is not None:
        missing = set(np.asarray(test_ids)) - set(sub["id"])
        extra = set(sub["id"]) - set(np.asarray(test_ids))
        if missing:
            problems.append(f"{len(missing)} test ids missing")
        if extra:
            problems.append(f"{len(extra)} ids not in test.csv")
    return problems


def label_distribution(sub: pd.DataFrame) -> dict[str, float]:
    return {lab: float((sub["label"] == lab).mean()) for lab in LABELS}
