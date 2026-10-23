"""Charts used in README.md, regenerated from the run outputs -> docs/figures/*.png

uv run python tools/make_readme_figures.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

import matplotlib.pyplot as plt  # noqa: E402

from build_final import MMBERT, load_members  # noqa: E402
from rmitguard import charts as C  # noqa: E402
from rmitguard.constants import LABELS  # noqa: E402
from rmitguard.metrics import macro_f1  # noqa: E402
from rmitguard.xling import apply_override  # noqa: E402

OUT = ROOT / "docs" / "figures"
LANGS = ["pt", "fr", "it", "en"]
ENSEMBLE, SINGLE = C.SERIES[0], C.SERIES[1]

# public leaderboard (30% of test), in submission order
SUBMISSIONS = [
    ("02", "5 mmbert-base (fold 0) + tuned offsets", 0.96915, "ensemble"),
    ("04", "+ xling, argmax", 0.97254, "ensemble"),
    ("distill", "1 mmbert-base, distilled, 100% train", 0.96514, "single"),
    ("10", "7 mmbert x5 folds + TF-IDF + Granite (pt)", 0.97530, "ensemble"),
    ("11", "Granite 4 folds", 0.97555, "ensemble"),
    ("12a", "Granite pt weight 0.45", 0.97599, "ensemble"),
    ("12b", "Granite only where unsure", 0.97644, "ensemble"),
    ("12c", "+ 2 models trained on 100% train", 0.97913, "ensemble"),
    ("13", "+ 6 models on 100% train, Granite seed 2", 0.97824, "ensemble"),
]
TOP10_CUTOFF = 0.97769  # 10th place at 14:34, 22/10
# private leaderboard (70% of test) of the three submissions selected for final scoring
PRIVATE = {"10": 0.98241, "12c": 0.98111, "distill": 0.97226}


def public_vs_private() -> None:
    names = [s for s in SUBMISSIONS if s[0] in PRIVATE]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    yy = np.arange(len(names))[::-1]
    for t, (sid, _desc, pub, kind) in zip(yy, names):
        prv = PRIVATE[sid]
        ax.hlines(t, pub, prv, color=C.GRID, lw=2, zorder=1)
        ax.scatter(pub, t, s=60, color=C.MUTED, edgecolor=C.SURFACE, lw=2, zorder=3)
        ax.scatter(
            prv,
            t,
            s=60,
            color=ENSEMBLE if kind == "ensemble" else SINGLE,
            edgecolor=C.SURFACE,
            lw=2,
            zorder=3,
        )
        ax.text(pub, t - 0.32, f"{pub:.5f}", ha="center", fontsize=7.5, color=C.INK2)
        ax.text(
            prv,
            t + 0.2,
            f"{prv:.5f}",
            ha="center",
            fontsize=8,
            color=C.INK,
            fontweight="bold" if sid == "10" else "normal",
        )
    ax.set_yticks(yy, [f"{s[0]}: {s[1]}" for s in names], fontsize=8)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.scatter([], [], s=50, color=C.MUTED, label="public (30%)")
    ax.scatter([], [], s=50, color=ENSEMBLE, label="private (70%, final ranking)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.set(title="Selected submissions: public vs private leaderboard", xlabel="macro F1")
    ax.grid(axis="y", visible=False)
    C._save(fig, OUT / "public_vs_private.png")


def lb_progress() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(SUBMISSIONS))
    scores = np.array([s[2] for s in SUBMISSIONS])
    ens = np.array([s[3] == "ensemble" for s in SUBMISSIONS])
    ax.plot(x[ens], scores[ens], color=ENSEMBLE, lw=2, zorder=2)
    ax.scatter(
        x[ens], scores[ens], s=60, color=ENSEMBLE, edgecolor=C.SURFACE, lw=2, zorder=3, label="ensemble"
    )
    ax.scatter(
        x[~ens], scores[~ens], s=60, color=SINGLE, edgecolor=C.SURFACE, lw=2, zorder=3, label="single model"
    )
    ax.axhline(TOP10_CUTOFF, color=C.MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(x[-1] + 0.35, TOP10_CUTOFF, "top-10 cut-off\n(14:34)", va="center", fontsize=8, color=C.INK2)
    for i in (0, 1, 2, 3, 7, 8):  # selective direct labels
        ax.annotate(
            f"{scores[i]:.5f}",
            (x[i], scores[i]),
            xytext=(0, 9 if i != 2 else -15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=C.INK,
            fontweight="bold" if i == 7 else "normal",
        )
    ax.set_xticks(x, [s[0] for s in SUBMISSIONS], fontsize=8.5)
    ax.set_xlim(-0.5, len(x) + 0.6)
    ax.set_ylim(0.962, 0.982)
    ax.set(
        title="Public leaderboard (30% of test) by submission",
        xlabel="submission (see README)",
        ylabel="macro F1",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    C._save(fig, OUT / "lb_progress.png")


def load() -> dict:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    y = train["label"].map({lab: i for i, lab in enumerate(LABELS)}).to_numpy()
    lang = train["language"].to_numpy()
    M = load_members(y, ["granite-2b-lora"])
    z = np.load(ROOT / "artifacts" / "xling" / "xling.npz")
    mm = sum(M[n]["oof"] for n in MMBERT) / len(MMBERT)
    e = 0.9 * mm + 0.1 * M["tfidf"]["oof"]
    g = M["granite-2b-lora"]["oof"]
    has_g = ~np.isnan(g).any(1)
    xl = apply_override(e, z["train_sim"], y[z["train_nn"]], 0.85)
    w = np.where((lang == "pt") & has_g, 0.4, 0.0)[:, None]
    final = apply_override((1 - w) * e + w * np.nan_to_num(g), z["train_sim"], y[z["train_nn"]], 0.85)
    best_single = max(MMBERT, key=lambda n: macro_f1(y, M[n]["oof"].argmax(1)))
    return dict(
        y=y, lang=lang, M=M, mm=mm, e=e, g=g, has_g=has_g, xl=xl, final=final, best_single=best_single
    )


def cv_ladder(d: dict) -> None:
    y, m = d["y"], d["has_g"]
    steps = [
        (f"best single model ({d['best_single']})", d["M"][d["best_single"]]["oof"]),
        ("7 mmbert-base members x 5 folds", d["mm"]),
        ("+ TF-IDF (10%)", d["e"]),
        ("+ xling (cross-lingual label lookup)", d["xl"]),
        ("+ Granite-2B on Portuguese rows (40%)", d["final"]),
    ]
    vals = [macro_f1(y[m], p[m].argmax(1)) for _, p in steps]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    yy = np.arange(len(steps))[::-1]
    ax.hlines(yy, 0.96, vals, color=C.GRID, lw=2, zorder=1)
    ax.scatter(vals, yy, s=70, color=ENSEMBLE, edgecolor=C.SURFACE, lw=2, zorder=3)
    for v, t in zip(vals, yy):
        ax.text(v + 0.0004, t, f"{v:.4f}", va="center", fontsize=8.5, color=C.INK)
    ax.set_yticks(yy, [s[0] for s in steps], fontsize=8.5)
    ax.set_xlim(0.96, 0.98)
    ax.set(
        title=f"Out-of-fold macro F1, step by step ({m.sum():,} train rows with a Granite prediction)",
        xlabel="macro F1 (argmax)",
    )
    ax.grid(axis="y", visible=False)
    C._save(fig, OUT / "cv_ladder.png")


def errors_by_language(d: dict) -> None:
    y, lang, m = d["y"], d["lang"], d["has_g"]
    single = d["M"][d["best_single"]]["oof"].argmax(1)
    final = d["final"].argmax(1)
    rows = [((lang == lg) & m) for lg in LANGS]
    a = [np.mean(single[r] != y[r]) * 100 for r in rows]
    b = [np.mean(final[r] != y[r]) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x, w = np.arange(len(LANGS)), 0.36
    ax.bar(x - w / 2 - 0.01, a, w, color=SINGLE, label=f"best single model ({d['best_single']})")
    ax.bar(x + w / 2 + 0.01, b, w, color=ENSEMBLE, label="final blend")
    for i in range(len(LANGS)):
        ax.text(x[i] - w / 2, a[i] + 0.08, f"{a[i]:.1f}%", ha="center", fontsize=8, color=C.INK)
        ax.text(x[i] + w / 2, b[i] + 0.08, f"{b[i]:.1f}%", ha="center", fontsize=8, color=C.INK)
    n = [int(r.sum()) for r in rows]
    ax.set_xticks(x, [f"{lg}\n{k:,} rows" for lg, k in zip(LANGS, n)])
    ax.set(title="Error rate by language (out-of-fold)", ylabel="% of rows misclassified")
    ax.legend(frameon=False, fontsize=8)
    C._save(fig, OUT / "errors_by_language.png")


def granite_by_language(d: dict) -> None:
    y, lang, m = d["y"], d["lang"], d["has_g"]
    g, e = d["g"], d["e"]
    rows = [((lang == lg) & m) for lg in LANGS]
    a = [macro_f1(y[r], e[r].argmax(1)) for r in rows]
    b = [macro_f1(y[r], np.nan_to_num(g)[r].argmax(1)) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    yy = np.arange(len(LANGS))[::-1]
    for t, va, vb in zip(yy, a, b):
        ax.hlines(t, min(va, vb), max(va, vb), color=C.GRID, lw=2, zorder=1)
    ax.scatter(
        a,
        yy,
        s=65,
        color=ENSEMBLE,
        edgecolor=C.SURFACE,
        lw=2,
        zorder=3,
        label="mmbert-base ensemble + TF-IDF",
    )
    ax.scatter(
        b,
        yy,
        s=65,
        color=C.SERIES[2],
        edgecolor=C.SURFACE,
        lw=2,
        zorder=3,
        label="Granite-2B (LoRA, 1 model/fold)",
    )
    for t, va, vb in zip(yy, a, b):
        ax.text(va, t + 0.22, f"{va:.3f}", ha="center", fontsize=7.5, color=C.INK2)
        ax.text(vb, t - 0.36, f"{vb:.3f}", ha="center", fontsize=7.5, color=C.INK2)
    ax.set_yticks(yy, LANGS)
    ax.set_ylim(-0.7, len(LANGS) - 0.4)
    ax.set(
        title="Macro F1 by language: Granite alone is weaker, closest on Portuguese",
        xlabel="macro F1 (out-of-fold)",
    )
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    C._save(fig, OUT / "granite_by_language.png")


def backbones() -> None:
    s = pd.read_csv(ROOT / "experiments" / "summary" / "summary.csv")
    s = s[(s["model"] == s["backbone"]) | (s["model"] == "tfidf")].copy()
    s["name"] = s["model"].replace({"granite-2b": "granite-2b-lora"})
    s = s.sort_values("macro_f1_argmax")
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(s) + 1.1))
    yy = np.arange(len(s))
    colors = [ENSEMBLE if n == "mmbert-base" else C.MUTED for n in s["name"]]
    ax.hlines(yy, 0.90, s["macro_f1_argmax"], color=C.GRID, lw=2, zorder=1)
    ax.scatter(s["macro_f1_argmax"], yy, s=55, color=colors, edgecolor=C.SURFACE, lw=2, zorder=3)
    for t, v in zip(yy, s["macro_f1_argmax"]):
        ax.text(v + 0.0012, t, f"{v:.4f}", va="center", fontsize=8, color=C.INK)
    ax.set_yticks(yy, s["name"], fontsize=8.5)
    ax.set_xlim(0.90, 0.975)
    ax.set(
        title="Benchmark: one model per backbone, fold 0 (2,800 rows), default settings",
        xlabel="macro F1 (argmax)",
    )
    ax.grid(axis="y", visible=False)
    C._save(fig, OUT / "backbones.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lb_progress()
    public_vs_private()
    backbones()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = load()
    cv_ladder(d)
    errors_by_language(d)
    granite_by_language(d)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
