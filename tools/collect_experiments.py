"""Gather every run (local artifacts/ + downloaded Kaggle outputs/) into experiments/<stage>/<run>/ with its
diagnostic charts and report, plus comparison charts and tables. Safe to re-run at any time.

    uv run python tools/collect_experiments.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from rmitguard import charts as C  # noqa: E402
from rmitguard.charts import plot_compare, plot_run, plot_training_health  # noqa: E402
from rmitguard.constants import LABELS  # noqa: E402
from rmitguard.ensemble import ensemble, find_runs  # noqa: E402

EXP = ROOT / "experiments"
FULL_SPLIT = "10a79e50dde8"
# Kaggle notebook tag -> experiment folder
STAGES = {
    "bench-a": "03-bench-fold0",
    "bench-b": "03-bench-fold0",
    "tune": "04-tune-fold0",
    "llm": "05-llm-fold0",
    "holdout": "06-holdout",
}
# first runs, made before the prediction-order bug was fixed: only their in-training curves are valid
DEBUG = {"xlmr-large-f0": "xlmr-large", "mmbert-base-f0": "mmbert-base", "wolf-defender-f0": "wolf-defender"}


TUNE_SUFFIXES = ("-ls", "-lang", "-cw0", "-ep2", "-ep", "-lr")


def stage_of(run_dir: Path) -> str | None:
    """Classify a run by what it is (not by which notebook produced it)."""
    rel = run_dir.relative_to(ROOT).as_posix()
    if rel.startswith("artifacts/"):
        return "01-tfidf-baseline"
    tag = rel.split("/")[1].removeprefix("rmit-sgs-gendata-").removesuffix("-c1")
    if tag in DEBUG:
        return "02-pipeline-debug"
    meta = json.loads((run_dir / "meta.json").read_text())
    name = meta["name"]
    if meta.get("holdout"):
        return "06-holdout"
    if name.startswith(("granite", "mistral")):
        return "05-llm-fold0"
    if name.endswith(TUNE_SUFFIXES):
        return "04-tune-fold0"
    if meta.get("folds_done") == [0] and meta.get("split_fp") == FULL_SPLIT and tag.startswith("bench"):
        return "03-bench-fold0"
    return "08-fold-runs"  # fold subsets of v1 / seeds; merged in 07-v1 and 09-ensembles


def copy_run(run_dir: Path, dest: Path, health_only: bool = False) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    if health_only:
        h = run_dir / "history.csv"
        if h.exists():
            hh = pd.read_csv(h).rename(
                columns=lambda c: (
                    c.replace("eval_", "eval_val_")
                    if c.startswith("eval_") and not c.startswith(("eval_val_", "eval_train_"))
                    else c
                )
            )
            plot_training_health(
                hh, f"{dest.name} — training health (in-training eval only)", dest / "training_health.png"
            )
        return
    try:
        plot_run(run_dir)
    except Exception as exc:  # a broken run must not stop the collection
        (dest / "ERROR.txt").write_text(f"charts failed: {exc}\n")
    if (run_dir / "charts").exists():
        for f in (run_dir / "charts").glob("*.png"):
            shutil.copy(f, dest / f.name)
    for f in ("report.md", "top_errors.csv", "meta.json"):
        if (run_dir / f).exists():
            shutil.copy(run_dir / f, dest / f)


def eda_charts() -> None:
    d = EXP / "00-eda"
    d.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "data" / "train.csv")
    train["n_chars"] = train["text"].str.len()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bins = np.logspace(1, np.log10(20000), 50)
    for lab in LABELS:
        ax.hist(
            train.loc[train["label"] == lab, "n_chars"],
            bins=bins,
            histtype="step",
            lw=2,
            color=C.CLASS_COLOR[lab],
            label=f"{lab} (median {int(train.loc[train['label'] == lab, 'n_chars'].median())})",
        )
    ax.set_xscale("log")
    ax.set(title="Prompt length by class (characters, log scale)", xlabel="characters", ylabel="prompts")
    ax.legend()
    C._save(fig, d / "length_by_class.png")
    oof = ROOT / "artifacts" / "tfidf" / "oof_predictions.csv"
    if oof.exists():
        p = pd.read_csv(oof)
        err = (p["label"] != p["pred"]).groupby([p["language"], p["label"]]).mean().unstack()[list(LABELS)]
        err = err.loc[["pt", "fr", "it", "en"]]
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.imshow(err.to_numpy(), cmap=C.BLUES, vmin=0, vmax=max(0.25, float(err.to_numpy().max())))
        ax.grid(False)
        ax.set_xticks(range(3), ["BENIGN", "JAILBREAK", "PROMPT_INJ."])
        ax.set_yticks(range(4), err.index)
        for i in range(err.shape[0]):
            for j in range(3):
                v = err.iat[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.1%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if v > 0.15 else C.INK,
                )
        ax.set(title="TF-IDF error rate by language × true class", xlabel="true class")
        C._save(fig, d / "tfidf_error_heatmap.png")
    if (ROOT / "reports" / "eda.md").exists():
        shutil.copy(ROOT / "reports" / "eda.md", d / "eda.md")


def table(df: pd.DataFrame) -> str:
    cols = {
        "model": "run",
        "macro_f1": "macro F1",
        "macro_f1_argmax": "argmax",
        "f1_BENIGN": "F1 B",
        "f1_JAILBREAK": "F1 J",
        "f1_PROMPT_INJECTION": "F1 PI",
        "pt_f1_JAILBREAK": "pt J",
        "en_f1_PROMPT_INJECTION": "en PI",
        "minutes_per_fold": "min/fold",
        "peak_vram_gb": "VRAM GB",
        "overfit_gap": "val−train loss",
        "best_val_f1_epoch": "best epoch",
    }
    return df[[c for c in cols if c in df]].rename(columns=cols).round(4).to_markdown(index=False)


def main() -> None:
    EXP.mkdir(exist_ok=True)
    eda_charts()
    runs = [
        d
        for d in find_runs([ROOT / "outputs", ROOT / "artifacts"])
        if "members" not in json.loads((d / "meta.json").read_text())
    ]
    by_stage: dict[str, list[Path]] = {}
    for d in runs:
        st = stage_of(d)
        if st is None:
            continue
        name = json.loads((d / "meta.json").read_text())["name"]
        dest = EXP / st / name
        copy_run(d, dest, health_only=(st == "02-pipeline-debug"))
        by_stage.setdefault(st, []).append(d)

    # holdout: merge the fold subsets that ran on the two GPUs
    if "06-holdout" in by_stage:
        ens = ensemble(by_stage["06-holdout"], EXP / "06-holdout" / "merged")
        plot_run(ens)
        for f in (ens / "charts").glob("*.png"):
            shutil.copy(f, ens / f.name)

    # comparison of everything scored on fold 0 of the full split (same 2,800 rows)
    fold0 = []
    for st in ("01-tfidf-baseline", "03-bench-fold0", "04-tune-fold0", "05-llm-fold0"):
        for d in by_stage.get(st, []):
            m = json.loads((d / "meta.json").read_text())
            if m.get("split_fp") == FULL_SPLIT and m.get("folds_done") == [0]:
                fold0.append(d)
    summary = None
    if fold0:
        summary = plot_compare(fold0, EXP / "summary", title="Fold 0 (2,800 rows)")

    lines = [
        "# Experiments",
        "",
        "Generated by `tools/collect_experiments.py` — every chart below is regenerated from the run artifacts.",
        "",
        "| folder | what |",
        "|---|---|",
        "| `00-eda/` | data facts: length by class, TF-IDF error heatmap, eda.md |",
        "| `01-tfidf-baseline/` | TF-IDF + logistic regression (CPU): 5-fold, fold 0, holdout 20% |",
        "| `02-pipeline-debug/` | first GPU runs, before the prediction-order fix (only in-training curves valid) |",
        "| `03-bench-fold0/` | every approved encoder, fold 0, default hyper-parameters |",
        "| `04-tune-fold0/` | variants of the two best encoders (epochs, lr, language tag, class weights) |",
        "| `05-llm-fold0/` | Granite-2B / Mistral-7B, 4-bit + LoRA |",
        "| `06-holdout/` | best encoder config, 20% locked holdout (private-LB estimate) |",
        "| `summary/` | comparison charts + `summary.csv` for every fold-0 run |",
        "",
    ]
    if summary is not None:
        lines += [
            "## Fold-0 comparison (same 2,800 validation rows)",
            "",
            table(summary),
            "",
            "![ranking](summary/ranking.png)",
            "![bottlenecks](summary/bottlenecks.png)",
            "![per class](summary/per_class.png)",
            "![cost](summary/cost_vs_score.png)",
            "![loss curves](summary/learning_curves_loss.png)",
            "![f1 curves](summary/learning_curves_f1.png)",
            "",
        ]
    (EXP / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"collected {len(runs)} runs into {EXP.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
