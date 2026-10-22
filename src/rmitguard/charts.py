"""Charts for one run (training health + error inspection) and for comparing runs.

    plot_run(run_dir)                  -> run_dir/charts/*.png + run_dir/top_errors.csv
    plot_compare(run_dirs, out_dir)    -> out_dir/*.png + out_dir/summary.csv

Colours: the validated categorical palette of the dataviz reference (fixed order, never cycled).
Classes always use slots 1-3 (BENIGN blue, JAILBREAK orange, PROMPT_INJECTION aqua), which are
colour-blind safe as a set. More than 8 models are never coloured individually: small multiples instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import LABELS
from .metrics import macro_f1

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLASS_COLOR = dict(zip(LABELS, SERIES[:3]))
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
)
LENGTH_BUCKETS = [
    (0, 100, "<100"),
    (100, 300, "100–299"),
    (300, 800, "300–799"),
    (800, 2000, "800–1999"),
    (2000, 10**9, "2000+"),
]  # characters

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.size": 9,
        "lines.linewidth": 2,
        "figure.dpi": 110,
    }
)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _bars(ax, labels, values, color=SERIES[0], fmt="{:.3f}", xlim=None):
    """Dot plot (not bars): honest even when the axis does not start at zero. Values in ink."""
    y = np.arange(len(labels))
    ax.hlines(y, xlim[0] if xlim else 0, values, color=GRID, lw=1, zorder=1)
    ax.scatter(values, y, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        if np.isfinite(v):
            ax.text(v, yi, "   " + fmt.format(v), va="center", fontsize=8, color=INK)
    if xlim:
        ax.set_xlim(*xlim)


def _f1_xlim(values) -> tuple[float, float]:
    v = np.asarray([x for x in values if np.isfinite(x)])
    lo = max(0.0, (v.min() if len(v) else 0.5) - 0.05)
    return (np.floor(lo * 20) / 20, 1.0 + (1.0 - lo) * 0.12)


# ------------------------------------------------------------------------------------------- one run
def load_predictions(run_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("holdout", "oof"):
        p = run_dir / f"{name}_predictions.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
    return out


def plot_training_health(history: pd.DataFrame, title: str, path: Path) -> None:
    """Loss (train vs probe vs val), validation F1 per class, gradient norm, learning rate — per epoch."""
    h = history.copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    (a_loss, a_f1), (a_grad, a_lr) = axes
    folds = sorted(h["fold"].dropna().unique()) if "fold" in h else [0]

    def by_epoch(col):
        d = h.dropna(subset=[col]) if col in h else pd.DataFrame(columns=["epoch", col])
        return d.groupby("epoch")[col].agg(["mean", "min", "max"])

    curves = [
        ("loss", "train loss (logged while training)", SERIES[0]),
        ("eval_train_loss", "train loss (fixed train sample, eval mode)", SERIES[1]),
        ("eval_val_loss", "validation loss", SERIES[2]),
    ]
    for col, label, color in curves:
        m = by_epoch(col)
        if len(m):
            a_loss.plot(
                m.index,
                m["mean"],
                color=color,
                label=label,
                marker="o" if col != "loss" else None,
                ms=4,
                lw=2 if col != "loss" else 1.2,
            )
            if len(folds) > 1:
                a_loss.fill_between(m.index, m["min"], m["max"], color=color, alpha=0.12, lw=0)
    a_loss.set(
        title="Loss — gap between orange and aqua = overfitting", xlabel="epoch", ylabel="cross-entropy"
    )
    a_loss.legend(loc="upper right")

    m = by_epoch("eval_val_macro_f1")
    if len(m):
        a_f1.plot(m.index, m["mean"], color=INK, lw=2.4, label="macro F1", marker="o", ms=4)
    for lab in LABELS:
        mm = by_epoch(f"eval_val_f1_{lab}")
        if len(mm):
            a_f1.plot(mm.index, mm["mean"], color=CLASS_COLOR[lab], lw=1.6, label=lab)
    a_f1.set(title="Validation F1 per class", xlabel="epoch", ylabel="F1 (argmax)")
    a_f1.legend(loc="lower right")

    for ax, col, ttl in (
        (a_grad, "grad_norm", "Gradient norm (spikes / NaN = unstable)"),
        (a_lr, "learning_rate", "Learning-rate schedule"),
    ):
        d = h.dropna(subset=[col]) if col in h else pd.DataFrame()
        for i, (k, g) in enumerate(d.groupby("fold") if len(d) else []):
            ax.plot(g["epoch"], g[col], color=SERIES[i % len(SERIES)], lw=1.2, label=f"fold {int(k)}")
        ax.set(title=ttl, xlabel="epoch")
        if len(folds) > 1:
            ax.legend()
    a_lr.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    _save(fig, path)


def plot_confusion(frames: dict[str, pd.DataFrame], title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(frames), figsize=(5.2 * len(frames), 4.4), squeeze=False)
    for ax, (name, df) in zip(axes[0], frames.items()):
        cm = pd.crosstab(
            pd.Categorical(df["label"], LABELS), pd.Categorical(df["pred"], LABELS), dropna=False
        )
        rate = cm.to_numpy() / np.maximum(cm.to_numpy().sum(1, keepdims=True), 1)
        ax.imshow(rate, cmap=BLUES, vmin=0, vmax=1)
        ax.grid(False)
        short = ["BENIGN", "JAILBR.", "P_INJ."]
        ax.set_xticks(range(3), short)
        ax.set_yticks(range(3), short)
        ax.set(
            xlabel="predicted",
            ylabel="true",
            title=f"{name}: {len(df)} rows, macro F1 "
            f"{macro_f1(df['label'].map(LABELS.index).to_numpy(), df['pred'].map(LABELS.index).to_numpy()):.4f}",
        )
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    f"{cm.iat[i, j]}\n{rate[i, j]:.1%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if rate[i, j] > 0.55 else INK,
                )
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    _save(fig, path)


def _f1_by(df: pd.DataFrame, key: str, order) -> pd.DataFrame:
    from sklearn.metrics import f1_score

    rows = []
    for v in order:
        d = df[df[key] == v]
        if len(d) == 0:
            continue
        y, p = d["label"].map(LABELS.index), d["pred"].map(LABELS.index)
        f = f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
        rows.append(dict(group=v, n=len(d), macro=f.mean(), **dict(zip(LABELS, f))))
    return pd.DataFrame(rows)


def plot_slices(df: pd.DataFrame, title: str, path: Path) -> None:
    """F1 per class by language and by prompt length — where the model breaks."""
    df = df.copy()
    df["length"] = pd.cut(
        df["n_chars"],
        [b[0] for b in LENGTH_BUCKETS] + [10**9],
        right=False,
        labels=[b[2] for b in LENGTH_BUCKETS],
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, key, order, ttl in (
        (axes[0], "language", ["pt", "fr", "it", "en"], "F1 by language"),
        (axes[1], "length", [b[2] for b in LENGTH_BUCKETS], "F1 by prompt length (chars)"),
    ):
        t = _f1_by(df, key, order)
        x = np.arange(len(t))
        w = 0.26
        for i, lab in enumerate(LABELS):
            ax.bar(
                x + (i - 1) * w,
                t[lab],
                width=w,
                color=CLASS_COLOR[lab],
                edgecolor=SURFACE,
                linewidth=2,
                label=lab,
            )
        ax.plot(x, t["macro"], color=INK, marker="D", ms=5, lw=0, label="macro F1")
        for xi, (v, n) in enumerate(zip(t["macro"], t["n"])):
            ax.text(xi, 1.005, f"{v:.3f}\nn={n}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(x, t["group"])
        ax.set_ylim(0, 1.12)
        ax.set(title=ttl, ylabel="F1")
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=4, bbox_to_anchor=(1, 1.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    _save(fig, path)


def plot_confidence(df: pd.DataFrame, title: str, path: Path) -> None:
    """Is the model's confidence honest? Histogram (right vs wrong) + reliability curve."""
    p = df[[f"p_{lab}" for lab in LABELS]].to_numpy()
    conf = p.max(1)
    correct = (df["label"] == np.asarray(LABELS)[p.argmax(1)]).to_numpy()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.linspace(1 / 3, 1, 21)
    a.hist(conf[correct], bins=bins, color=SERIES[0], label=f"correct ({correct.sum()})", edgecolor=SURFACE)
    a.hist(conf[~correct], bins=bins, color=SERIES[1], label=f"wrong ({(~correct).sum()})", edgecolor=SURFACE)
    a.set_yscale("log")
    a.set(title="Confidence of the top class (log scale)", xlabel="max probability", ylabel="rows")
    a.legend()
    edges = np.linspace(1 / 3, 1, 9)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, len(edges) - 2)
    acc = [correct[idx == i].mean() if (idx == i).any() else np.nan for i in range(len(edges) - 1)]
    mid = [conf[idx == i].mean() if (idx == i).any() else np.nan for i in range(len(edges) - 1)]
    b.plot([1 / 3, 1], [1 / 3, 1], color=MUTED, lw=1, ls="--", label="perfectly calibrated")
    b.plot(mid, acc, color=SERIES[0], marker="o", ms=6, label="model")
    b.set(title="Reliability: accuracy at each confidence level", xlabel="mean confidence", ylabel="accuracy")
    b.legend(loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    _save(fig, path)


def top_errors(df: pd.DataFrame, n: int = 60) -> pd.DataFrame:
    p = df[[f"p_{lab}" for lab in LABELS]]
    wrong = df[df["label"] != df["pred"]].copy()
    wrong["confidence"] = p.loc[wrong.index].max(axis=1)
    return wrong.sort_values("confidence", ascending=False).head(n)


def plot_run(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    name = meta["name"]
    charts = run_dir / "charts"
    made = []
    hist_path = run_dir / "history.csv"
    if hist_path.exists() and hist_path.stat().st_size > 5:
        h = pd.read_csv(hist_path)
        if "eval_val_loss" in h or "loss" in h:
            plot_training_health(h, f"{name} — training health", charts / "training_health.png")
            made.append(charts / "training_health.png")
    preds = load_predictions(run_dir)
    if preds:
        plot_confusion(preds, f"{name} — confusion (row-normalised)", charts / "confusion.png")
        main = preds.get("holdout", preds.get("oof"))
        which = "holdout" if "holdout" in preds else "out-of-fold"
        plot_slices(main, f"{name} — where it breaks ({which})", charts / "slices.png")
        plot_confidence(main, f"{name} — confidence ({which})", charts / "confidence.png")
        top_errors(main).to_csv(run_dir / "top_errors.csv", index=False)
        made += [charts / "confusion.png", charts / "slices.png", charts / "confidence.png"]
    return made


# ------------------------------------------------------------------------------------------- compare
def summarize(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text())
    s, ho = meta.get("scores") or {}, meta.get("holdout")
    src = ho or s
    pc = pd.DataFrame(src.get("per_class", {}))
    pl = pd.DataFrame(src.get("per_language", {}))
    folds = meta.get("fold_stats", [])
    row = dict(
        model=meta["name"],
        backbone=meta.get("model"),
        macro_f1=(ho or {}).get("macro_f1", s.get("macro_f1_tuned_crossfit", s.get("macro_f1_tuned"))),
        macro_f1_argmax=src.get("macro_f1_argmax"),
        minutes_per_fold=float(np.mean([f.get("minutes", np.nan) for f in folds])) if folds else np.nan,
        peak_vram_gb=float(np.max([f.get("peak_vram_gb", 0) for f in folds])) if folds else np.nan,
        truncated=float(np.mean([f.get("truncated", np.nan) for f in folds])) if folds else np.nan,
        folds=len(folds),
    )
    for lab in LABELS:
        row[f"f1_{lab}"] = pc.get("f1", {}).get(lab, np.nan) if len(pc) else np.nan
    if len(pl):
        row["pt_f1_JAILBREAK"] = pl.get("f1_JAILBREAK", {}).get("pt", np.nan)
        row["en_f1_PROMPT_INJECTION"] = pl.get("f1_PROMPT_INJECTION", {}).get("en", np.nan)
    h = run_dir / "history.csv"
    if h.exists() and h.stat().st_size > 5:
        hh = pd.read_csv(h)
        if "eval_val_loss" in hh and "eval_train_loss" in hh:
            last = hh["epoch"].max()
            v = hh.loc[hh["epoch"] == last, "eval_val_loss"].dropna()
            t = hh.loc[hh["epoch"] == last, "eval_train_loss"].dropna()
            row["final_val_loss"] = float(v.mean()) if len(v) else np.nan
            row["overfit_gap"] = float(v.mean() - t.mean()) if len(v) and len(t) else np.nan
            row["best_val_f1_epoch"] = (
                float(hh.loc[hh["eval_val_macro_f1"].idxmax(), "epoch"])
                if "eval_val_macro_f1" in hh and hh["eval_val_macro_f1"].notna().any()
                else np.nan
            )
    return row


def plot_compare(
    run_dirs: list[str | Path], out_dir: str | Path, title: str = "Benchmark — fold 0"
) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = [Path(d) for d in run_dirs]
    t = (
        pd.DataFrame([summarize(d) for d in runs])
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )
    t.to_csv(out / "summary.csv", index=False)

    # 1. ranking
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(t) + 1.4))
    _bars(ax, t["model"], t["macro_f1"], xlim=_f1_xlim(t["macro_f1"]))
    ax.set(title=f"{title}: macro F1 (decision offsets tuned)", xlabel="macro F1")
    _save(fig, out / "ranking.png")

    # 2. per-class F1 — small multiples, one panel per class
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.42 * len(t) + 1.6), sharey=True)
    for ax, lab in zip(axes, LABELS):
        _bars(ax, t["model"], t[f"f1_{lab}"], color=CLASS_COLOR[lab], xlim=_f1_xlim(t[f"f1_{lab}"]))
        ax.set(title=f"F1 {lab}", xlabel="F1")
    _save(fig, out / "per_class.png")

    # 3. the two bottlenecks found in EDA
    cols = [c for c in ("pt_f1_JAILBREAK", "en_f1_PROMPT_INJECTION") if c in t]
    if cols:
        fig, axes = plt.subplots(
            1, len(cols), figsize=(7.5 * len(cols), 0.42 * len(t) + 1.6), sharey=True, squeeze=False
        )
        names = {
            "pt_f1_JAILBREAK": ("Bottleneck 1: JAILBREAK F1 in Portuguese", CLASS_COLOR["JAILBREAK"]),
            "en_f1_PROMPT_INJECTION": (
                "Bottleneck 2: PROMPT_INJECTION F1 in English",
                CLASS_COLOR["PROMPT_INJECTION"],
            ),
        }
        for ax, c in zip(axes[0], cols):
            _bars(ax, t["model"], t[c], color=names[c][1], xlim=_f1_xlim(t[c]))
            ax.set(title=names[c][0], xlabel="F1")
        _save(fig, out / "bottlenecks.png")

    # 4. cost vs score
    d = t.dropna(subset=["minutes_per_fold", "macro_f1"])
    if len(d):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(
            d["minutes_per_fold"],
            d["macro_f1"],
            s=70,
            color=SERIES[0],
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
        )
        for _, r in d.iterrows():
            ax.annotate(
                r["model"],
                (r["minutes_per_fold"], r["macro_f1"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )
        ax.set_xscale("log")
        ax.set(
            title="Cost vs score: minutes per fold (log) against macro F1",
            xlabel="minutes per fold",
            ylabel="macro F1",
        )
        _save(fig, out / "cost_vs_score.png")

    # 5. learning curves — small multiples (one panel per model, same axes)
    hist = {}
    for d_ in runs:
        h = d_ / "history.csv"
        if h.exists() and h.stat().st_size > 5:
            hh = pd.read_csv(h)
            if "eval_val_loss" in hh:
                hist[json.loads((d_ / "meta.json").read_text())["name"]] = hh
    if hist:
        order = [m for m in t["model"] if m in hist]
        ncol = min(4, len(order))
        nrow = int(np.ceil(len(order) / ncol))
        for metric, fname, ylab in (
            ("loss", "learning_curves_loss.png", "loss"),
            ("f1", "learning_curves_f1.png", "validation macro F1"),
        ):
            fig, axes = plt.subplots(
                nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), sharex=False, sharey=True, squeeze=False
            )
            for ax in axes.flat[len(order) :]:
                ax.set_visible(False)
            for ax, m in zip(axes.flat, order):
                hh = hist[m]
                if metric == "loss":
                    for col, color, lab in (
                        ("eval_train_loss", SERIES[1], "train (probe)"),
                        ("eval_val_loss", SERIES[2], "validation"),
                    ):
                        g = hh.dropna(subset=[col]).groupby("epoch")[col].mean()
                        ax.plot(g.index, g.values, color=color, marker="o", ms=3, label=lab)
                else:
                    g = hh.dropna(subset=["eval_val_macro_f1"]).groupby("epoch")["eval_val_macro_f1"].mean()
                    ax.plot(g.index, g.values, color=SERIES[0], marker="o", ms=3)
                ax.set_title(m, fontsize=9)
                ax.set_xlabel("epoch")
            axes[0][0].set_ylabel(ylab)
            if metric == "loss":
                axes[0][0].legend()
            fig.suptitle(
                f"{title}: {ylab} per epoch (one panel per model)",
                x=0.01,
                ha="left",
                fontsize=13,
                fontweight="bold",
                color=INK,
            )
            _save(fig, out / fname)
    return t
