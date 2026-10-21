"""Loading the competition CSVs and building leak-free, reproducible CV folds."""

from __future__ import annotations

import glob
import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .constants import COMPETITION, L2I, LABELS

KAGGLE_DATA = f"/kaggle/input/competitions/{COMPETITION}"


def find_data_dir(explicit: str | Path | None = None) -> Path:
    """Explicit path > $RMITGUARD_DATA > ./data > Kaggle mount (any depth, model folders excluded)."""
    candidates = [explicit, os.environ.get("RMITGUARD_DATA"), "data", KAGGLE_DATA]
    for c in candidates:
        if c and (Path(c) / "train.csv").exists():
            return Path(c)
    hits = [p for p in glob.glob("/kaggle/input/**/train.csv", recursive=True) if "/models/" not in p]
    hits.sort(key=lambda p: ("rmit" not in p.lower(), p))
    if hits:
        return Path(hits[0]).parent
    raise FileNotFoundError(
        "train.csv not found. Locally: `make data` (you must have joined the competition on Kaggle). "
        "On Kaggle: Add Input -> Competitions -> RMIT HACKATHON 2026."
    )


@dataclass
class CompetitionData:
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def y(self) -> np.ndarray:
        return self.train["label"].map(L2I).to_numpy()


def load_competition(data_dir: str | Path | None = None) -> CompetitionData:
    d = find_data_dir(data_dir)
    train, test = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    for df in (train, test):
        df["text"] = df["text"].fillna("").astype(str)
        if "language" not in df:
            df["language"] = "unk"
    bad = sorted(set(train["label"]) - set(LABELS))
    if bad:
        raise ValueError(f"unexpected labels in train.csv: {bad}")
    return CompetitionData(train=train, test=test)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.md5(normalize(text).encode("utf-8")).hexdigest()


def make_folds(train: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    """Fold id per row.

    Stratified on label x language so every fold sees the rare PROMPT_INJECTION class in every language,
    and grouped on the normalised text so duplicated prompts never sit on both sides of a split
    (duplicates across folds inflate CV and make it useless for picking models).
    """
    strata = train["label"].astype(str) + "|" + train["language"].astype(str)
    groups = train["text"].map(text_hash).to_numpy()
    folds = np.full(len(train), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(splitter.split(train, strata, groups)):
        folds[va] = k
    return folds


def split_fingerprint(folds: np.ndarray) -> str:
    """Runs can only be merged (ensembles, fold subsets from teammates) when this matches."""
    return hashlib.md5(folds.astype(np.int64).tobytes()).hexdigest()[:12]
