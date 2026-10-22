"""Rebuild the final submission (public LB 0.97913) from the downloaded run outputs.

Recipe (every step chosen on out-of-fold predictions, see README):
  mmbert part   mean of 7 mmbert-base members x 5 folds + 2 models trained on 100% of train (test only)
  + TF-IDF      10% of the probability mass
  + Granite-2B  40% weight on Portuguese rows only (folds whose OOF macro F1 < 0.9 are dropped: they diverged)
  + xling       rows whose cross-lingual nearest train neighbour has cosine >= 0.85 take its label
  -> argmax (no tuned class offsets)

    uv run python tools/build_final.py --out submissions/RMIT_SGS_GenData_C1.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rmitguard.constants import LABELS  # noqa: E402
from rmitguard.ensemble import find_runs, load_run, merge_fold_subsets  # noqa: E402
from rmitguard.metrics import macro_f1  # noqa: E402
from rmitguard.submission import make_submission, validate_submission  # noqa: E402
from rmitguard.xling import apply_override  # noqa: E402

FULL_SPLIT = "10a79e50dde8"
# first GPU runs, made before the prediction-order fix: their OOF is wrong
DEBUG = ("mmbert-base-f0-c1", "wolf-defender-f0-c1", "xlmr-large-f0-c1")
MMBERT = [
    "mmbert-base",
    "mmbert-base-cw0",
    "mmbert-base-lang",
    "mmbert-base-lang-s2",
    "mmbert-base-ls",
    "mmbert-base-s2",
    "mmbert-base-s4",
]
FULL_TRAIN = ["mmbert-base-full", "mmbert-base-distill-full"]


def load_members(y: np.ndarray, granite: list[str]) -> dict[str, dict]:
    """Every full-split run under outputs/ (+ the CPU TF-IDF run), fold subsets merged by name."""
    runs = []
    for d in find_runs([ROOT / "outputs", ROOT / "artifacts" / "tfidf"]):
        if any(k in str(d) for k in DEBUG):
            continue
        r = load_run(d)
        m = r["meta"]
        if "members" in m or m.get("holdout") or m.get("split_fp") != FULL_SPLIT:
            continue
        if m["name"] in granite:  # an LLM fold that diverged is worse than no fold
            ok = ~np.isnan(r["oof"]).any(1)
            f1 = macro_f1(y[ok], r["oof"][ok].argmax(1)) if ok.any() else 0.0
            if f1 < 0.9 or np.isnan(r["test"]).any():
                print(f"drop {d.relative_to(ROOT)}: OOF macro F1 {f1:.3f}")
                continue
        runs.append(r)
    return merge_fold_subsets(runs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submissions/RMIT_SGS_GenData_C1.csv")
    ap.add_argument("--granite", nargs="+", default=["granite-2b-lora"], help="Granite run names (seeds)")
    ap.add_argument("--full-train", nargs="*", default=FULL_TRAIN, help="100%%-train runs added on test")
    ap.add_argument("--granite-pt", type=float, default=0.4)
    ap.add_argument("--tfidf", type=float, default=0.1)
    ap.add_argument("--xling", type=float, default=0.85)
    args = ap.parse_args()

    train, test = pd.read_csv(ROOT / "data" / "train.csv"), pd.read_csv(ROOT / "data" / "test.csv")
    y = train["label"].map({lab: i for i, lab in enumerate(LABELS)}).to_numpy()
    pt, test_pt = (train["language"] == "pt").to_numpy(), (test["language"] == "pt").to_numpy()

    M = load_members(y, args.granite)
    missing = [n for n in [*MMBERT, "tfidf", *args.granite, *args.full_train] if n not in M]
    if missing:
        raise SystemExit(f"missing runs: {missing} (download the Kaggle outputs first)")

    oof_mm = sum(M[n]["oof"] for n in MMBERT) / len(MMBERT)
    test_mm = (sum(M[n]["test"] for n in MMBERT) + sum(M[n]["test"] for n in args.full_train)) / (
        len(MMBERT) + len(args.full_train)
    )
    oof_e = (1 - args.tfidf) * oof_mm + args.tfidf * M["tfidf"]["oof"]
    test_e = (1 - args.tfidf) * test_mm + args.tfidf * M["tfidf"]["test"]
    with warnings.catch_warnings():  # rows outside every Granite fold stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        oof_g = np.nanmean(np.stack([M[n]["oof"] for n in args.granite]), axis=0)
    test_g = np.mean(np.stack([M[n]["test"] for n in args.granite]), axis=0)
    has_g = ~np.isnan(oof_g).any(1)

    z = np.load(ROOT / "artifacts" / "xling" / "xling.npz")

    def blend(e, g, is_pt, sim, nn_label):
        w = np.where(is_pt, args.granite_pt, 0.0)[:, None]
        return apply_override((1 - w) * e + w * np.nan_to_num(g), sim, nn_label, args.xling)

    oof = blend(oof_e, oof_g, pt & has_g, z["train_sim"], y[z["train_nn"]])
    print(
        f"OOF macro F1 on the {has_g.sum()} rows with a Granite prediction: "
        f"{macro_f1(y[has_g], oof[has_g].argmax(1)):.4f} "
        f"(Granite folds {sorted({f for n in args.granite for f in M[n]['folds_done']})})"
    )
    pred = blend(test_e, test_g, test_pt, z["test_sim"], y[z["test_nn"]]).argmax(1)
    sub = make_submission(test["id"], pred)
    problems = validate_submission(sub, test["id"], expected_rows=len(test))
    if problems:
        raise SystemExit(f"invalid submission: {problems}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(
        f"wrote {out.relative_to(ROOT)}  {json.dumps(sub['label'].value_counts(normalize=True).round(3).to_dict())}"
    )


if __name__ == "__main__":
    main()
