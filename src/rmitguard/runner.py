"""Run one experiment: K-fold CV -> OOF + averaged test probabilities -> report + submission.

Artifacts are rewritten after every fold, so a Kaggle session that hits its time limit still leaves
usable output, and teammates can each run a subset of folds and merge them with `rmitguard ensemble`.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .constants import LABELS
from .data import CompetitionData, load_competition, make_folds, split_fingerprint
from .metrics import bootstrap_ci, confusion_table, macro_f1, per_class_table, per_language_table
from .models import get_trainer
from .postprocess import cross_fit_bias_score, fit_bias, predict_with_bias
from .submission import label_distribution, make_submission, validate_submission


def subsample(data: CompetitionData, limit: int, seed: int) -> CompetitionData:
    """Stratified subset for smoke tests (`--limit 300`)."""
    train = data.train.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(limit * len(g) // len(data.train), 10)), random_state=seed)
    )
    return CompetitionData(
        train=train.reset_index(drop=True), test=data.test.head(limit).reset_index(drop=True)
    )


def score_oof(oof: np.ndarray, y: np.ndarray, folds: np.ndarray, language: np.ndarray) -> dict:
    """Scores on the rows that have OOF predictions (all rows once every fold has run)."""
    m = ~np.isnan(oof).any(axis=1)
    if not m.any():
        return {}
    probs, yy = oof[m], y[m]
    pred_argmax = probs.argmax(1)
    bias, tuned = fit_bias(probs, yy)
    pred = predict_with_bias(probs, bias)
    out = dict(
        coverage=float(m.mean()),
        macro_f1_argmax=macro_f1(yy, pred_argmax),
        macro_f1_tuned=tuned,
        bias=bias.tolist(),
        public_lb_ci95=bootstrap_ci(yy, pred, n=500, sample_size=1800),
        per_class=per_class_table(yy, pred),
        per_language=per_language_table(yy, pred, language[m]),
        confusion=confusion_table(yy, pred),
    )
    if len(np.unique(folds[m])) >= 2:
        out["macro_f1_tuned_crossfit"] = cross_fit_bias_score(probs, yy, folds[m])
    return out


def write_report(
    run_dir: Path,
    title: str,
    header: list[str],
    scores: dict,
    sub: pd.DataFrame | None,
    fold_stats: list[dict],
) -> None:
    lines = [f"# {title}", "", *(f"- {h}" for h in header), ""]
    if scores:
        lines += [
            "| metric | value |",
            "|---|---|",
            f"| OOF coverage | {scores['coverage']:.1%} |",
            f"| macro F1 (argmax) | {scores['macro_f1_argmax']:.4f} |",
            f"| macro F1 (tuned offsets, in-sample) | {scores['macro_f1_tuned']:.4f} |",
        ]
        if "macro_f1_tuned_crossfit" in scores:
            lines.append(
                f"| **macro F1 (tuned, cross-fitted) ← trust this** | **{scores['macro_f1_tuned_crossfit']:.4f}** |"
            )
        lo, hi = scores["public_lb_ci95"]
        lines += [
            f"| public-LB-sized noise (95% CI, 1800 rows) | {lo:.4f} – {hi:.4f} |",
            f"| offsets (log-prob) | {np.round(scores['bias'], 2).tolist()} |",
            "",
            "## Per class",
            scores["per_class"].round(4).to_markdown(),
            "",
            "## Per language",
            scores["per_language"].round(4).to_markdown(),
            "",
            "## Confusion (OOF, tuned)",
            scores["confusion"].to_markdown(),
            "",
        ]
    if sub is not None:
        dist = label_distribution(sub)
        lines += ["## Test prediction distribution", *(f"- {k}: {v:.1%}" for k, v in dist.items()), ""]
    if fold_stats:
        lines += ["## Folds", pd.DataFrame(fold_stats).round(4).to_markdown(index=False)]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _jsonable(scores: dict) -> dict:
    return {k: (v.round(5).to_dict() if isinstance(v, pd.DataFrame) else v) for k, v in scores.items()}


def run_experiment(
    cfg: ExperimentConfig,
    data_dir: str | None = None,
    out_dir: str | Path = "artifacts",
    limit: int | None = None,
) -> Path:
    t_start = time.time()
    data = load_competition(data_dir)
    if limit:
        data = subsample(data, limit, cfg.seed)
    train, test, y = data.train, data.test, data.y
    folds = make_folds(train, cfg.n_folds, cfg.seed)
    split_fp = split_fingerprint(folds)
    language = train["language"].astype(str).to_numpy()

    run_dir = Path(out_dir) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    oof = np.full((len(train), len(LABELS)), np.nan, dtype=np.float32)
    test_sum = np.zeros((len(test), len(LABELS)), dtype=np.float64)
    fold_stats, history, done = [], [], []
    train_fold = get_trainer(cfg.kind)
    print(f"[{cfg.name}] {len(train)} train / {len(test)} test | folds {cfg.fold_ids()} | split {split_fp}")

    for k in cfg.fold_ids():
        elapsed_h = (time.time() - t_start) / 3600
        if fold_stats and elapsed_h + fold_stats[-1]["minutes"] / 60 > cfg.time_budget_hours:
            print(f"⏭️ stopping before fold {k}: time budget {cfg.time_budget_hours}h would be exceeded")
            break
        tr, va = folds != k, folds == k
        val_p, test_p, info = train_fold(cfg, train[tr], y[tr], train[va], y[va], test, seed=cfg.seed + k)
        oof[va] = val_p
        test_sum += test_p
        done.append(k)
        history += [dict(fold=k, **row) for row in info.pop("history", [])]
        fold_stats.append(
            dict(
                fold=k,
                n_val=int(va.sum()),
                macro_f1=macro_f1(y[va], val_p.argmax(1)),
                minutes=info.get("minutes", (time.time() - t_start) / 60),
                **{k2: v for k2, v in info.items() if k2 != "minutes"},
            )
        )
        print(f"[{cfg.name}] fold {k}: macro F1 (argmax) = {fold_stats[-1]['macro_f1']:.4f}")

        # persist after every fold
        test_mean = (test_sum / len(done)).astype(np.float32)
        scores = score_oof(oof, y, folds, language)
        sub = make_submission(test["id"], predict_with_bias(test_mean, np.asarray(scores["bias"])))
        problems = validate_submission(sub, test["id"], expected_rows=len(test))
        if problems:
            raise RuntimeError(f"invalid submission: {problems}")
        meta = dict(
            name=cfg.name,
            model=cfg.model,
            hf_id=cfg.hf_id,
            config=cfg.to_dict(),
            split_fp=split_fp,
            n_train=len(train),
            n_test=len(test),
            folds_done=done,
            n_test_models=len(done),
            fold_stats=fold_stats,
            scores=_jsonable(scores),
            updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        np.save(run_dir / "oof.npy", oof)
        np.save(run_dir / "test.npy", test_mean)
        np.save(run_dir / "folds.npy", folds)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        sub.to_csv(run_dir / "submission.csv", index=False)
        write_report(
            run_dir,
            cfg.name,
            [
                f"model: `{cfg.model}` ({cfg.hf_id or 'local diagnostic, not an approved backbone'}) · split `{split_fp}`",
                f"folds done: {done} / {cfg.n_folds} · updated {meta['updated_at']}",
            ],
            scores,
            sub,
            fold_stats,
        )

    print((run_dir / "report.md").read_text(encoding="utf-8"))
    return run_dir
