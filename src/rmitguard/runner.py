"""Run one experiment: K-fold CV -> OOF + averaged test probabilities -> report + submission.

Two protocols:
- full    (holdout: 0)   5-fold CV on all of train.csv; used for the models we submit.
- holdout (holdout: 0.2) 20% of train is locked away as a private-leaderboard stand-in; CV runs on the other
                         80% (dev). Decision offsets are fitted on dev OOF only, then the holdout is scored once.
                         Used to benchmark backbones on the same 2,800 rows.

Artifacts are rewritten after every fold, so a Kaggle session that hits its time limit still leaves
usable output, and teammates can each run a subset of folds and merge them with `rmitguard ensemble`.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .constants import LABELS
from .data import CompetitionData, load_competition, make_split, split_fingerprint
from .metrics import bootstrap_ci, confusion_table, macro_f1, per_class_table, per_language_table
from .models import get_trainer
from .postprocess import cross_fit_bias_score, fit_bias, predict_with_bias
from .submission import label_distribution, make_submission, validate_submission


def subsample(data: CompetitionData, limit: int, seed: int) -> CompetitionData:
    """Stratified subset for smoke tests (`--limit 300`)."""
    # pandas 3 drops the grouping column inside groupby().apply, so pick indices per class instead
    idx = [
        i
        for _, g in data.train.groupby("label")
        for i in g.sample(min(len(g), max(limit * len(g) // len(data.train), 10)), random_state=seed).index
    ]
    return CompetitionData(
        train=data.train.loc[sorted(idx)].reset_index(drop=True),
        test=data.test.head(limit).reset_index(drop=True),
    )


def run_name(cfg: ExperimentConfig) -> str:
    return f"{cfg.name}-ho" if cfg.holdout else cfg.name


def score_oof(oof: np.ndarray, y: np.ndarray, folds: np.ndarray, language: np.ndarray) -> dict:
    """Scores on the rows that have OOF predictions (holdout rows never do)."""
    m = ~np.isnan(oof).any(axis=1)
    if not m.any():
        return {}
    probs, yy = oof[m], y[m]
    bias, tuned = fit_bias(probs, yy)
    pred = predict_with_bias(probs, bias)
    out = dict(
        coverage=float(m.sum() / (folds >= 0).sum()),
        macro_f1_argmax=macro_f1(yy, probs.argmax(1)),
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


def score_holdout(probs: np.ndarray, y: np.ndarray, language: np.ndarray, bias: np.ndarray) -> dict:
    """The holdout is scored with offsets learnt on dev only — an honest private-LB estimate."""
    pred = predict_with_bias(probs, bias)
    return dict(
        n=len(y),
        macro_f1_argmax=macro_f1(y, probs.argmax(1)),
        macro_f1=macro_f1(y, pred),
        ci95=bootstrap_ci(y, pred, n=1000),
        public_lb_ci95=bootstrap_ci(y, pred, n=500, sample_size=1800),
        per_class=per_class_table(y, pred),
        per_language=per_language_table(y, pred, language),
        confusion=confusion_table(y, pred),
    )


def write_report(
    run_dir: Path,
    title: str,
    header: list[str],
    scores: dict,
    sub: pd.DataFrame | None,
    fold_stats: list[dict],
    holdout: dict | None = None,
) -> None:
    lines = [f"# {title}", "", *(f"- {h}" for h in header), ""]
    if holdout:
        lo, hi = holdout["ci95"]
        lines += [
            f"## Holdout ({holdout['n']} rows, never trained on, offsets from dev)",
            "| metric | value |",
            "|---|---|",
            f"| **holdout macro F1 ← benchmark score** | **{holdout['macro_f1']:.4f}** (95% CI {lo:.4f} – {hi:.4f}) |",
            f"| holdout macro F1 (argmax, no offsets) | {holdout['macro_f1_argmax']:.4f} |",
            "",
            holdout["per_class"].round(4).to_markdown(),
            "",
            holdout["per_language"].round(4).to_markdown(),
            "",
            holdout["confusion"].to_markdown(),
            "",
        ]
    if scores:
        lines += [
            "## Dev out-of-fold" if holdout else "## Out-of-fold (CV)",
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
            "### Per class",
            scores["per_class"].round(4).to_markdown(),
            "",
            "### Per language",
            scores["per_language"].round(4).to_markdown(),
            "",
            "### Confusion (OOF, tuned)",
            scores["confusion"].to_markdown(),
            "",
        ]
    if sub is not None:
        dist = label_distribution(sub)
        lines += ["## Test prediction distribution", *(f"- {k}: {v:.1%}" for k, v in dist.items()), ""]
    if fold_stats:
        lines += ["## Folds", pd.DataFrame(fold_stats).round(4).to_markdown(index=False)]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _jsonable(scores: dict | None) -> dict | None:
    if scores is None:
        return None
    return {k: (v.round(5).to_dict() if isinstance(v, pd.DataFrame) else v) for k, v in scores.items()}


def predictions_frame(df: pd.DataFrame, probs: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Per-row predictions for error analysis (`charts.py`, notebooks)."""
    out = df[["id", "language", "label"]].copy()
    out["pred"] = np.asarray(LABELS)[pred]
    for i, lab in enumerate(LABELS):
        out[f"p_{lab}"] = probs[:, i].round(5)
    out["n_chars"] = df["text"].str.len().to_numpy()
    out["text"] = df["text"].str.slice(0, 400).to_numpy()
    return out


def run_experiment(
    cfg: ExperimentConfig,
    data_dir: str | None = None,
    out_dir: str | Path = "artifacts",
    limit: int | None = None,
) -> Path:
    t_start = time.time()
    data = load_competition(data_dir)
    if limit:
        data = subsample(data, limit, cfg.split_seed)
    train, test, y = data.train, data.test, data.y
    folds = make_split(train, cfg.n_folds, cfg.split_seed, cfg.holdout)
    split_fp = split_fingerprint(folds)
    language = train["language"].astype(str).to_numpy()
    hold = folds == -1
    hold_df = train[hold]
    # holdout rows are predicted together with the test set (no change to the model interface)
    predict_df = pd.concat([hold_df, test], ignore_index=True) if hold.any() else test
    n_hold = int(hold.sum())

    name = run_name(cfg)
    run_dir = Path(out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    oof = np.full((len(train), len(LABELS)), np.nan, dtype=np.float32)
    pred_sum = np.zeros((len(predict_df), len(LABELS)), dtype=np.float64)
    fold_stats, history, done = [], [], []
    train_fold = get_trainer(cfg.kind)
    print(
        f"[{name}] {int((folds >= 0).sum())} dev / {n_hold} holdout / {len(test)} test | "
        f"folds {cfg.fold_ids()} | split {split_fp}"
    )

    for k in cfg.fold_ids():
        elapsed_h = (time.time() - t_start) / 3600
        if fold_stats and elapsed_h + fold_stats[-1]["minutes"] / 60 > cfg.time_budget_hours:
            print(f"⏭️ stopping before fold {k}: time budget {cfg.time_budget_hours}h would be exceeded")
            break
        tr, va = (folds != k) & (folds >= 0), folds == k
        if cfg.full_fit:
            tr = folds >= 0
        y_tr = y[tr]
        if cfg.soft_targets:  # (n_rows, n_classes) float targets instead of class ids
            teacher = np.load(cfg.soft_targets)[: len(train)]
            y_tr = cfg.distill_alpha * np.eye(len(LABELS))[y[tr]] + (1 - cfg.distill_alpha) * teacher[tr]
        val_p, pred_p, info = train_fold(
            cfg, train[tr], y_tr, train[va], y[va], predict_df, seed=cfg.seed + k
        )
        oof[va] = val_p
        pred_sum += pred_p
        done.append(k)
        history += [dict(fold=k, **row) for row in info.pop("history", [])]
        fold_stats.append(
            dict(
                fold=k,
                n_train=int(tr.sum()),
                n_val=int(va.sum()),
                macro_f1=macro_f1(y[va], val_p.argmax(1)),
                minutes=info.get("minutes", (time.time() - t_start) / 60),
                **{k2: v for k2, v in info.items() if k2 != "minutes"},
            )
        )
        print(f"[{name}] fold {k}: macro F1 (argmax) = {fold_stats[-1]['macro_f1']:.4f}")

        # persist after every fold
        pred_mean = (pred_sum / len(done)).astype(np.float32)
        hold_p, test_mean = pred_mean[:n_hold], pred_mean[n_hold:]
        scores = score_oof(oof, y, folds, language)
        # argmax: OOF-tuned offsets lose to it out-of-fold (see ensemble.py), and are meaningless with full_fit
        bias = np.zeros(len(LABELS))
        holdout = score_holdout(hold_p, y[hold], language[hold], bias) if n_hold else None
        sub = make_submission(test["id"], predict_with_bias(test_mean, bias))
        problems = validate_submission(sub, test["id"], expected_rows=len(test))
        if problems:
            raise RuntimeError(f"invalid submission: {problems}")
        meta = dict(
            name=name,
            model=cfg.model,
            hf_id=cfg.hf_id,
            config=cfg.to_dict(),
            split_fp=split_fp,
            n_train=len(train),
            n_holdout=n_hold,
            n_test=len(test),
            folds_done=done,
            n_test_models=len(done),
            fold_stats=fold_stats,
            scores=_jsonable(scores),
            holdout=_jsonable(holdout),
            elapsed_minutes=round((time.time() - t_start) / 60, 2),
            updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        np.save(run_dir / "oof.npy", oof)
        np.save(run_dir / "test.npy", test_mean)
        np.save(run_dir / "folds.npy", folds)
        if n_hold:
            np.save(run_dir / "holdout.npy", hold_p)
            predictions_frame(hold_df, hold_p, predict_with_bias(hold_p, bias)).to_csv(
                run_dir / "holdout_predictions.csv", index=False
            )
        m = ~np.isnan(oof).any(axis=1)
        predictions_frame(train[m], oof[m], predict_with_bias(oof[m], bias)).to_csv(
            run_dir / "oof_predictions.csv", index=False
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        sub.to_csv(run_dir / "submission.csv", index=False)
        write_report(
            run_dir,
            name,
            [
                f"model: `{cfg.model}` ({cfg.hf_id or 'local diagnostic, not an approved backbone'}) · split `{split_fp}`",
                f"folds done: {done} / {cfg.n_folds} · {meta['elapsed_minutes']} min · updated {meta['updated_at']}",
            ],
            scores,
            sub,
            fold_stats,
            holdout,
        )

    try:  # charts must never cost us a finished run
        from .charts import plot_run

        plot_run(run_dir)
    except Exception:
        print(f"⚠️ charts failed:\n{traceback.format_exc()}")
    print((run_dir / "report.md").read_text(encoding="utf-8"))
    return run_dir
