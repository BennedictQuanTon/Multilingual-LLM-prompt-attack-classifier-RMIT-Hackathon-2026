"""Collect every downloaded run, redraw its charts and build the comparison report.

uv run python tools/bench_report.py                      # outputs/ + artifacts/tfidf -> reports/bench/
uv run python tools/bench_report.py outputs artifacts/tfidf --out reports/bench --split 10a79e50dde8
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rmitguard.charts import plot_compare, plot_run  # noqa: E402
from rmitguard.ensemble import find_runs  # noqa: E402

COLS = {
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=["outputs", "artifacts/tfidf"])
    ap.add_argument("--out", default="reports/bench")
    ap.add_argument("--split", default=None, help="keep only runs with this split fingerprint")
    args = ap.parse_args()
    out = ROOT / args.out
    runs = []
    for d in find_runs([ROOT / r for r in args.roots]):
        meta = json.loads((d / "meta.json").read_text())
        if "members" in meta or (args.split and meta.get("split_fp") != args.split):
            continue
        runs.append(d)
    if not runs:
        raise SystemExit("no runs found")
    for d in runs:
        plot_run(d)
        dest = out / "runs" / json.loads((d / "meta.json").read_text())["name"]
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(d / "charts", dest)
        for f in ("report.md", "top_errors.csv"):
            if (d / f).exists():
                shutil.copy(d / f, dest / f)
    t = plot_compare(runs, out)
    table = t[[c for c in COLS if c in t]].rename(columns=COLS)
    lines = [
        "# Benchmark",
        "",
        f"{len(runs)} runs · split `{args.split or 'any'}`",
        "",
        table.round(4).to_markdown(index=False),
        "",
        "![ranking](ranking.png)",
        "![per class](per_class.png)",
        "![bottlenecks](bottlenecks.png)",
        "![cost vs score](cost_vs_score.png)",
        "![loss](learning_curves_loss.png)",
        "![f1](learning_curves_f1.png)",
        "",
        "Per-run charts: `runs/<name>/`.",
    ]
    (out / "BENCH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table.round(4).to_string(index=False))
    print(f"\n-> {out.relative_to(ROOT)}/BENCH.md")


if __name__ == "__main__":
    main()
