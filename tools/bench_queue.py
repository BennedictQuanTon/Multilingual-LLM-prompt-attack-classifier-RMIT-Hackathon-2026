"""Benchmark queue: push fold-0 notebooks one after another on one Kaggle account, at most N at a time,
and download each output as soon as it finishes.

    uv run python tools/bench_queue.py configs/mmbert-base.yaml configs/xlmr-base.yaml ... [--parallel 2]

Prints one line per event (pushed / status change / downloaded); state in outputs/bench_queue.json.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "outputs" / "bench_queue.json"
DONE = ("complete", "error", "cancel")


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M} {msg}", flush=True)


def kaggle(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["uv", "run", "kaggle", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=check,
        env={**__import__("os").environ, "KAGGLE_DISABLE_WARN": "1"},
    )


def build(config: str, folds: str) -> tuple[Path, str]:
    out = subprocess.run(
        ["uv", "run", "python", "tools/build_kernel.py", "train", config, "--folds", folds],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    path = ROOT / out.split()[1]
    return path, json.loads((path / "kernel-metadata.json").read_text())["id"]


def status(kernel: str) -> str:
    out = kaggle("kernels", "status", kernel).stdout.strip().splitlines()
    m = re.search(r'status "(?:KernelWorkerStatus\.)?(\w+)"', out[-1] if out else "")
    return m.group(1).lower() if m else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--watch", nargs="*", default=[], help="kernels already running to wait for / download")
    args = ap.parse_args()

    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    for k in args.watch:
        state.setdefault(k, {"config": "?", "status": "running"})
    pending = list(args.configs)
    while pending or any(v["status"] not in DONE for v in state.values()):
        for kernel, v in state.items():
            if v["status"] in DONE:
                continue
            s = status(kernel)
            if s != v["status"]:
                log(f"{kernel}: {s}")
                v["status"] = s
                if s in DONE:
                    dest = ROOT / "outputs" / kernel.split("/")[1]
                    dest.mkdir(parents=True, exist_ok=True)
                    r = kaggle("kernels", "output", kernel, "-p", str(dest))
                    log(
                        f"{kernel}: output -> outputs/{dest.name} ({'ok' if r.returncode == 0 else r.stderr[-200:]})"
                    )
        running = sum(v["status"] not in DONE for v in state.values())
        while pending and running < args.parallel:
            cfg = pending.pop(0)
            path, kernel = build(cfg, args.folds)
            r = kaggle("kernels", "push", "-p", str(path))
            if r.returncode != 0 or "error" in (r.stdout + r.stderr).lower():
                log(f"push {cfg} failed: {(r.stdout + r.stderr).strip()[-300:]} — retrying later")
                pending.insert(0, cfg)
                break
            state[kernel] = {"config": cfg, "status": "pushed", "pushed_at": datetime.now().isoformat()}
            log(f"pushed {cfg} -> {kernel}")
            running += 1
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state, indent=2))
        time.sleep(60)
    log("queue finished")


if __name__ == "__main__":
    sys.exit(main())
