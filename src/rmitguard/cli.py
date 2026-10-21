"""Command line entry point: `rmitguard <command>` (or `python -m rmitguard <command>`)."""

from __future__ import annotations

import argparse
import sys

from .constants import L2I, LABELS, N_TEST_ROWS


def _folds(value: str | None) -> list[int] | None:
    return None if value in (None, "", "all") else [int(x) for x in value.split(",")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rmitguard", description=__doc__)
    parser.add_argument(
        "--data", default=None, help="folder with train.csv / test.csv (default: auto-detect)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("eda", help="write reports/eda.md")
    p.add_argument("--out", default="reports/eda.md")

    p = sub.add_parser("train", help="run one experiment config with K-fold CV")
    p.add_argument("config")
    p.add_argument("--folds", default=None, help="comma-separated subset, e.g. 0,1 (default: all)")
    p.add_argument("--limit", type=int, default=None, help="train on a small stratified subset (smoke test)")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--name", default=None, help="override the experiment name")

    p = sub.add_parser("ensemble", help="merge fold subsets / runs that share a CV split")
    p.add_argument("runs", nargs="+", help="run directories or roots to search for them")
    p.add_argument("--out", default="artifacts/ensemble")
    p.add_argument("--weights", default=None, help="name=weight,name=weight")
    p.add_argument("--allow-multi-backbone", action="store_true")

    p = sub.add_parser("validate", help="check a submission CSV before spending a submission on it")
    p.add_argument("submission")

    sub.add_parser("labels", help="print the label order used everywhere")

    args = parser.parse_args(argv)

    if args.cmd == "eda":
        from .eda import eda_report

        print(eda_report(args.data, args.out))
    elif args.cmd == "train":
        from .config import load_config
        from .runner import run_experiment

        cfg = load_config(args.config, folds=_folds(args.folds), name=args.name)
        run_experiment(cfg, data_dir=args.data, out_dir=args.out, limit=args.limit)
    elif args.cmd == "ensemble":
        from .ensemble import ensemble

        weights = (
            dict((k, float(v)) for k, v in (kv.split("=") for kv in args.weights.split(",")))
            if args.weights
            else None
        )
        ensemble(
            args.runs,
            out_dir=args.out,
            data_dir=args.data,
            weights=weights,
            allow_multi_backbone=args.allow_multi_backbone,
        )
    elif args.cmd == "validate":
        import pandas as pd

        from .data import find_data_dir
        from .submission import label_distribution, validate_submission

        try:
            test_ids = pd.read_csv(find_data_dir(args.data) / "test.csv")["id"]
        except FileNotFoundError:
            test_ids = None
            print("⚠️ test.csv not found; checking format only")
        problems = validate_submission(
            args.submission, test_ids, expected_rows=len(test_ids) if test_ids is not None else N_TEST_ROWS
        )
        if problems:
            print("❌ NOT safe to submit:\n- " + "\n- ".join(problems))
            return 1
        print(
            "✅ valid —", {k: f"{v:.1%}" for k, v in label_distribution(pd.read_csv(args.submission)).items()}
        )
    elif args.cmd == "labels":
        print({lab: L2I[lab] for lab in LABELS})
    return 0


if __name__ == "__main__":
    sys.exit(main())
