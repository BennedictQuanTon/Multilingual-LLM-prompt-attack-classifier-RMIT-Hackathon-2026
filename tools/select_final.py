"""Pick the final ensemble on the full 5-fold OOF (14k rows) instead of fold 0 alone (2.8k rows).

Prints: every member's OOF macro F1, the equal-weight ensemble of all 5-fold members, leave-one-out,
the xling threshold on top of it, and for fold-0-only members (e.g. Granite) the gain on fold 0.
Decisions are argmax (no tuned offsets). Members: approved backbones + the CPU TF-IDF model (team decision).

    uv run python tools/select_final.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rmitguard.constants import APPROVED_MODELS, LABELS  # noqa: E402
from rmitguard.ensemble import find_runs, load_run, merge_fold_subsets  # noqa: E402
from rmitguard.metrics import macro_f1  # noqa: E402
from rmitguard.xling import apply_override  # noqa: E402

FULL_SPLIT = "10a79e50dde8"
# first GPU runs, made before the prediction-order fix: their OOF is wrong
DEBUG = ("mmbert-base-f0-c1", "wolf-defender-f0-c1", "xlmr-large-f0-c1")


def load_members() -> dict[str, dict]:
    runs = []
    for d in find_runs([ROOT / "outputs", ROOT / "artifacts" / "tfidf"]):
        if any(k in str(d) for k in DEBUG):
            continue
        r = load_run(d)
        m = r["meta"]
        if "members" in m or m.get("holdout") or m.get("split_fp") != FULL_SPLIT:
            continue
        if m["model"] not in APPROVED_MODELS and m["model"] != "tfidf":
            continue
        runs.append(r)
    return merge_fold_subsets(runs)


def f1(y, p, mask=None):
    m = slice(None) if mask is None else mask
    return round(macro_f1(y[m], p[m].argmax(1)), 4)


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    y = train["label"].map({lab: i for i, lab in enumerate(LABELS)}).to_numpy()
    pt = train["language"].to_numpy() == "pt"
    z = np.load(ROOT / "artifacts" / "xling" / "xling.npz")
    members = load_members()
    full = {n: m for n, m in members.items() if m["folds_done"] == [0, 1, 2, 3, 4]}
    fold0 = {n: m for n, m in members.items() if 0 in m["folds_done"] and n not in full}

    print("== members with 5 folds (OOF on 14,000 rows)")
    for n, m in sorted(full.items(), key=lambda kv: -macro_f1(y, kv[1]["oof"].argmax(1))):
        print(f"  {n:28s} {f1(y, m['oof'])}  pt {f1(y, m['oof'], pt)}")
    if not full:
        return
    names = sorted(n for n in full if n != "tfidf")
    ens = sum(full[n]["oof"] for n in names) / len(names)
    print(f"\n== equal-weight ensemble of {len(names)} transformers: {f1(y, ens)}  pt {f1(y, ens, pt)}")
    for n in names:  # leave-one-out
        rest = [k for k in names if k != n]
        if rest:
            e = sum(full[k]["oof"] for k in rest) / len(rest)
            print(f"  without {n:24s} {f1(y, e)}")
    if "tfidf" in full:  # a different kind of model: its weight is tuned on its own, not 1/n
        print("\n== + TF-IDF at weight w")
        scores = {}
        for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            e = (1 - w) * ens + w * full["tfidf"]["oof"]
            scores[w] = macro_f1(y, e.argmax(1))
            print(f"  w {w:.2f}: {f1(y, e)}  pt {f1(y, e, pt)}")
        w = max(scores, key=scores.get)
        ens = (1 - w) * ens + w * full["tfidf"]["oof"]
        print(f"  -> TF-IDF weight {w}")

    print("\n== xling on the ensemble (train neighbours from other folds)")
    nnl = y[z["train_nn"]]
    for t in (None, 0.78, 0.8, 0.85, 0.9):
        e = ens if t is None else apply_override(ens, z["train_sim"], nnl, t)
        print(
            f"  threshold {t}: {f1(y, e)}  test rows overridden {0 if t is None else int((z['test_sim'] >= t).sum())}"
        )

    f0 = np.load(next(iter(full.values()))["sources"][0] + "/folds.npy") == 0
    if fold0:
        print("\n== fold-0-only members, added to the ensemble (fold 0, 2,800 rows)")
        print(f"  ensemble alone: {f1(y, ens, f0)}  pt {f1(y, ens, f0 & pt)}")
        for n, m in fold0.items():
            print(f"  {n:28s} alone {f1(y, m['oof'], f0)}  pt {f1(y, m['oof'], f0 & pt)}")
            for w in (0.2, 0.33, 0.5, 0.67):
                e = (1 - w) * ens + w * np.nan_to_num(m["oof"])
                print(f"    weight {w:.2f}: {f1(y, e, f0)}  pt {f1(y, e, f0 & pt)}")


if __name__ == "__main__":
    main()
