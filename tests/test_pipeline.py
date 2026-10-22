"""End-to-end on synthetic data with the CPU model: train -> artifacts -> fold-subset merge -> submission."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rmitguard.cli import main as cli
from rmitguard.config import ExperimentConfig
from rmitguard.ensemble import ensemble
from rmitguard.runner import run_experiment
from rmitguard.submission import validate_submission

ROOT = Path(__file__).resolve().parent.parent


def test_full_cv_run(data_dir, tmp_path):
    run = run_experiment(ExperimentConfig(name="tfidf", model="tfidf", n_folds=3), str(data_dir), tmp_path)
    meta = json.loads((run / "meta.json").read_text())
    assert meta["folds_done"] == [0, 1, 2]
    assert meta["scores"]["macro_f1_tuned_crossfit"] > 0.9  # the synthetic task is easy
    assert not np.isnan(np.load(run / "oof.npy")).any()
    test = pd.read_csv(data_dir / "test.csv")
    assert validate_submission(run / "submission.csv", test["id"], expected_rows=len(test)) == []
    assert (run / "report.md").read_text().count("Per language") == 1


def test_teammates_fold_subsets_merge_into_full_run(data_dir, tmp_path):
    for folds in ([0], [1, 2]):
        cfg = ExperimentConfig(name="tfidf", model="tfidf", n_folds=3, folds=folds)
        run_experiment(cfg, str(data_dir), tmp_path / f"member{folds[0]}")
    out = ensemble([tmp_path], tmp_path / "ens", data_dir=str(data_dir))
    meta = json.loads((out / "meta.json").read_text())
    assert meta["members"]["tfidf"]["folds_done"] == [0, 1, 2]
    assert meta["scores"]["coverage"] == 1.0
    # a second ensemble over the same root must ignore the first ensemble's output
    ensemble([tmp_path], tmp_path / "ens2", data_dir=str(data_dir))


def test_seed_variant_shares_split_and_merges(data_dir, tmp_path):
    run_experiment(ExperimentConfig(name="s42", model="tfidf", n_folds=3), str(data_dir), tmp_path)
    run_experiment(ExperimentConfig(name="s7", model="tfidf", n_folds=3, seed=7), str(data_dir), tmp_path)
    meta = json.loads(
        (ensemble([tmp_path], tmp_path / "ens", data_dir=str(data_dir)) / "meta.json").read_text()
    )
    assert set(meta["members"]) == {"s42", "s7"}


def test_ensemble_refuses_mixed_splits(data_dir, tmp_path):
    run_experiment(ExperimentConfig(name="a", model="tfidf", n_folds=3), str(data_dir), tmp_path)
    run_experiment(
        ExperimentConfig(name="b", model="tfidf", n_folds=3, split_seed=7), str(data_dir), tmp_path
    )
    with pytest.raises(ValueError, match="different CV splits"):
        ensemble([tmp_path], tmp_path / "ens", data_dir=str(data_dir))


def test_cli_train_and_validate(data_dir, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("name: cli\nmodel: tfidf\nn_folds: 3\n")
    assert cli(["--data", str(data_dir), "train", str(cfg), "--folds", "0", "--out", str(tmp_path)]) == 0
    assert cli(["--data", str(data_dir), "validate", str(tmp_path / "cli" / "submission.csv")]) == 0


def test_eda_report(data_dir, tmp_path):
    from rmitguard.eda import eda_report

    text = eda_report(str(data_dir), tmp_path / "eda.md")
    assert "CONFLICTING" in text and "PROMPT_INJECTION" in text


def _build(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_kernel.py"), *args, "--username", "tester"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_build_kernel_train_notebook():
    r = _build("train", "configs/xlmr-large.yaml", "--folds", "0,1")
    assert r.returncode == 0, r.stderr
    out = ROOT / "kaggle" / "build" / "rmit-sgs-gendata-xlmr-large-f01-c1"
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["title"] == "RMIT_SGS_GenData-xlmr-large-f01_C1"
    assert meta["id"] == "tester/rmit-sgs-gendata-xlmr-large-f01-c1"
    assert meta["competition_sources"] == ["rmit-hackathon-2026"]
    assert meta["model_sources"] == ["aisuko/xlm-roberta-large/Transformers/default/1"]
    nb = json.loads((out / "notebook.ipynb").read_text())
    sources = ["".join(c["source"]) for c in nb["cells"]]
    assert "FacebookAI/xlm-roberta-large" in sources[0]  # rules: state the chosen model at the top
    written = {
        s.splitlines()[0].split()[1]: s.split("\n", 1)[1] for s in sources if s.startswith("%%writefile")
    }
    for path, body in written.items():
        assert (ROOT / path).read_text().rstrip("\n") == body
    for s in sources:  # every plain code cell must be valid Python
        if not s.startswith("%%") and nb["cells"][sources.index(s)]["cell_type"] == "code":
            compile(s, "<cell>", "exec")


def test_build_kernel_final_name_and_tfidf_guard():
    r = _build("train", "configs/xlmr-base.yaml", "--final")
    assert r.returncode == 0, r.stderr
    meta = json.loads((ROOT / "kaggle/build/rmit-sgs-gendata-c1/kernel-metadata.json").read_text())
    assert meta["title"] == "RMIT_SGS_GenData_C1"
    assert _build("train", "configs/tfidf.yaml").returncode != 0


def test_holdout_protocol_scores_locked_rows_and_merges(data_dir, tmp_path):
    from rmitguard.data import load_competition, make_split

    train = load_competition(data_dir).train
    folds = make_split(train, 3, 42, holdout=0.2)
    assert abs((folds == -1).mean() - 0.2) < 0.03
    assert set(train.loc[folds == -1, "label"]) == {"BENIGN", "JAILBREAK", "PROMPT_INJECTION"}
    for part in ([0], [1, 2]):
        cfg = ExperimentConfig(name="t", model="tfidf", n_folds=3, folds=part, holdout=0.2)
        run = run_experiment(cfg, str(data_dir), tmp_path / f"m{part[0]}")
        assert run.name == "t-ho" and (run / "holdout.npy").exists()
        assert (run / "charts" / "confusion.png").exists() and (run / "top_errors.csv").exists()
    meta = json.loads(
        (ensemble([tmp_path], tmp_path / "ens", data_dir=str(data_dir)) / "meta.json").read_text()
    )
    assert meta["members"]["t-ho"]["folds_done"] == [0, 1, 2]
    assert meta["holdout"]["n"] == int((folds == -1).sum()) and meta["holdout"]["macro_f1"] > 0.9
