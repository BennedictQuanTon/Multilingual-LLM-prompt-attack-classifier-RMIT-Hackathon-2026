"""Experiment configuration: one YAML file in configs/ = one experiment."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .constants import APPROVED_MODELS


@dataclass
class ExperimentConfig:
    name: str
    model: str  # key of APPROVED_MODELS, or "tfidf" (local diagnostic only, not an approved backbone)
    seed: int = 42  # training seed: change it for extra ensemble members
    split_seed: int = 42  # CV split: keep it fixed so every run of the team can be merged
    n_folds: int = 5
    folds: list[int] | None = None  # None = all folds; a subset lets teammates split the work
    # Tokenisation: long prompts keep head + tail (role-play set-up at the start, real request at the end).
    max_len: int = 512
    head_len: int = 384
    prepend_language: bool = False
    # Optimisation
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 16
    grad_accum: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    class_weight: str = "sqrt"  # none | sqrt | balanced
    label_smoothing: float = 0.0
    grad_ckpt: bool = False
    fp16: bool = True
    # LLM route only (kind == "llm"): 4-bit base + LoRA adapters + classification head
    lora: dict[str, Any] = field(default_factory=lambda: {"r": 16, "alpha": 32, "dropout": 0.05})
    # TF-IDF only
    tfidf: dict[str, Any] = field(default_factory=lambda: {"C": 8.0})
    # Stop starting new folds when the next one would not finish inside Kaggle's 12 h limit.
    time_budget_hours: float = 11.0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.model != "tfidf" and self.model not in APPROVED_MODELS:
            raise ValueError(
                f"model {self.model!r} is not approved; choose from {sorted(APPROVED_MODELS)} or 'tfidf'"
            )
        if self.class_weight not in {"none", "sqrt", "balanced"}:
            raise ValueError(f"class_weight must be none|sqrt|balanced, got {self.class_weight!r}")
        if self.folds is not None and any(not 0 <= k < self.n_folds for k in self.folds):
            raise ValueError(f"folds {self.folds} out of range for n_folds={self.n_folds}")
        if not 0 < self.head_len <= self.max_len:
            raise ValueError("need 0 < head_len <= max_len")

    @property
    def kind(self) -> str:
        return "tfidf" if self.model == "tfidf" else APPROVED_MODELS[self.model][2]

    @property
    def hf_id(self) -> str | None:
        return APPROVED_MODELS[self.model][0] if self.model in APPROVED_MODELS else None

    @property
    def kaggle_model(self) -> str | None:
        return APPROVED_MODELS[self.model][1] if self.model in APPROVED_MODELS else None

    def fold_ids(self) -> list[int]:
        return list(range(self.n_folds)) if self.folds is None else sorted(self.folds)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown config keys {unknown} (typo?)")
        return cls(**data)


def load_config(path: str | Path, **overrides: Any) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    return ExperimentConfig.from_dict(data)


def load_config_text(text: str, **overrides: Any) -> ExperimentConfig:
    data = yaml.safe_load(text) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    return ExperimentConfig.from_dict(data)
