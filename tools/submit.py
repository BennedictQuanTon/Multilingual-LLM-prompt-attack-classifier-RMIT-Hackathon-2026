"""Guarded submission: validate, show what is about to be spent, log it, and only then submit.

python tools/submit.py --file artifacts/xlmr-large/submission.csv -m "xlmr-large 5 folds"          # dry run
python tools/submit.py --kernel user/rmit-sgs-gendata-c1 --version 3 -m "final" --yes             # submit
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from rmitguard.constants import COMPETITION, MAX_DAILY_SUBMISSIONS, N_TEST_ROWS  # noqa: E402
from rmitguard.data import find_data_dir  # noqa: E402
from rmitguard.submission import label_distribution, validate_submission  # noqa: E402

LOG = ROOT / "experiments" / "submissions.csv"


def kaggle_cmd() -> list[str]:
    if shutil.which("kaggle"):
        return ["kaggle"]
    if shutil.which("uvx"):
        return ["uvx", "kaggle"]
    raise SystemExit("kaggle CLI not found: `uv sync --extra dev` or `pip install kaggle`")


def used_today() -> int | None:
    try:
        out = subprocess.run(
            [*kaggle_cmd(), "competitions", "submissions", COMPETITION, "--csv"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except Exception as exc:
        print(f"(could not list submissions: {exc.__class__.__name__})")
        return None
    rows = list(csv.DictReader(io.StringIO(out[out.find("fileName") :] if "fileName" in out else out)))
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return sum(1 for r in rows if str(r.get("date", "")).startswith(today))


def fetch_kernel_output(kernel: str, dest: Path) -> Path:
    subprocess.run([*kaggle_cmd(), "kernels", "output", kernel, "-p", str(dest)], check=True)
    hits = [dest / "submission.csv", *dest.rglob("submission.csv")]
    for h in hits:
        if h.exists():
            return h
    raise SystemExit(f"no submission.csv in the output of {kernel}")


def cv_of(sub_path: Path) -> str:
    meta = sub_path.parent / "meta.json"
    if not meta.exists():
        return ""
    s = json.loads(meta.read_text()).get("scores", {})
    v = s.get("macro_f1_tuned_crossfit", s.get("macro_f1_tuned"))
    return f"{v:.4f}" if isinstance(v, float) else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path)
    src.add_argument("--kernel", help="owner/slug of the notebook whose output is submitted")
    ap.add_argument("--version", help="kernel version to submit (with --kernel)")
    ap.add_argument("-m", "--message", required=True)
    ap.add_argument("--yes", action="store_true", help="actually submit (default: dry run)")
    args = ap.parse_args()

    sub_path = args.file or fetch_kernel_output(args.kernel, Path(tempfile.mkdtemp(prefix="kernel_out_")))
    try:
        test_ids = pd.read_csv(find_data_dir() / "test.csv")["id"]
    except FileNotFoundError:
        test_ids = None
        print("⚠️ test.csv not found locally: checking format only")
    problems = validate_submission(
        sub_path, test_ids, expected_rows=len(test_ids) if test_ids is not None else N_TEST_ROWS
    )
    if problems:
        print("❌ NOT submitting:\n- " + "\n- ".join(problems))
        return 1
    dist = label_distribution(pd.read_csv(sub_path))
    n_today = used_today()
    print(
        f"file: {sub_path}\nlabels: { {k: f'{v:.1%}' for k, v in dist.items()} }\nCV (from meta.json): {cv_of(sub_path) or 'n/a'}"
    )
    print(
        f"submissions used today (UTC): {n_today if n_today is not None else '?'} / {MAX_DAILY_SUBMISSIONS}"
    )
    if n_today is not None and n_today >= MAX_DAILY_SUBMISSIONS:
        print("❌ daily limit reached")
        return 1
    if not args.yes:
        print("dry run — add --yes to submit")
        return 0

    cmd = [*kaggle_cmd(), "competitions", "submit", COMPETITION, "-m", args.message]
    if args.kernel:
        cmd += ["-k", args.kernel, "-f", "submission.csv"] + (["-v", args.version] if args.version else [])
    else:
        cmd += ["-f", str(sub_path)]
    subprocess.run(cmd, check=True)

    new = not LOG.exists()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc_time", "kernel", "version", "file", "cv_macro_f1", "message", "public_lb"])
        w.writerow(
            [
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                args.kernel or "",
                args.version or "",
                str(sub_path) if args.file else "",
                cv_of(sub_path),
                args.message,
                "",
            ]
        )
    print(f"✅ submitted and logged in {LOG.relative_to(ROOT)} — fill in public_lb when the score appears")
    return 0


if __name__ == "__main__":
    sys.exit(main())
