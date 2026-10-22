"""Overnight orchestrator: one Kaggle GPU session at a time, two models per session (one per T4).

Stages
  bench-a / bench-b  every approved encoder on fold 0 of the full split
  tune               hyper-parameter variants of the two best encoders (fold 0), chosen automatically
  llm                Granite-2B and Mistral-7B with 4-bit + LoRA (fold 0)
  holdout            best encoder config, 20% holdout locked, 5 folds on dev split over the two GPUs

Each finished stage is downloaded to outputs/<slug>/ and the experiment folder is refreshed.
Log: outputs/overnight.log · state: outputs/overnight.json. Never submits anything.

    uv run python -u tools/overnight.py > outputs/overnight.log 2>&1
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src")]
from bench_queue import kaggle, log, status  # noqa: E402

STATE = ROOT / "outputs" / "overnight.json"
SWEEP = ROOT / "configs" / "sweep"
ENCODERS_A = ["wolf-defender", "xlmr-base", "mmbert-base", "deberta-pi-v2"]
ENCODERS_B = ["mmbert-small", "distilbert-mbert", "xlmr-large", "rembert"]
LLMS = ["granite-2b-lora", "mistral-7b-lora"]
WAIT_FOR = [
    "longquanton/rmit-sgs-gendata-mmbert-base-f0-c1",
    "longquanton/rmit-sgs-gendata-wolf-defender-f0-c1",
]


def cfg_path(name: str) -> str:
    p = ROOT / "configs" / f"{name}.yaml"
    return str(p.relative_to(ROOT)) if p.exists() else str((SWEEP / f"{name}.yaml").relative_to(ROOT))


def run_stage(tag: str, jobs: list[str], holdout: float | None = None) -> Path | None:
    """Build a bench notebook for `jobs` (config paths, optionally with @folds), push it, wait, download."""
    cmd = ["uv", "run", "python", "tools/build_kernel.py", "bench", *jobs, "--tag", tag]
    if holdout:
        cmd += ["--holdout", str(holdout)]
    else:
        cmd += ["--folds", "0"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if out.returncode:
        log(f"[{tag}] build failed: {out.stderr[-400:]}")
        return None
    path = ROOT / out.stdout.split()[1]
    kernel = json.loads((path / "kernel-metadata.json").read_text())["id"]
    for attempt in range(30):
        r = kaggle("kernels", "push", "-p", str(path))
        text = (r.stdout + r.stderr).strip()
        if r.returncode == 0 and "error" not in text.lower():
            break
        log(f"[{tag}] push attempt {attempt + 1} failed: {text[-200:]}")
        time.sleep(120)
    else:
        return None
    log(f"[{tag}] pushed {kernel}: {', '.join(jobs)}")
    last, t0 = "", time.time()
    while True:
        s = status(kernel)
        if s != last:
            log(f"[{tag}] {s} ({(time.time() - t0) / 60:.0f} min)")
            last = s
        if s in ("complete", "error", "cancelacknowledged", "cancelled"):
            break
        time.sleep(60)
    dest = ROOT / "outputs" / kernel.split("/")[1]
    dest.mkdir(parents=True, exist_ok=True)
    r = kaggle("kernels", "output", kernel, "-p", str(dest))
    log(f"[{tag}] output -> outputs/{dest.name} ({'ok' if r.returncode == 0 else r.stderr[-200:]})")
    refresh()
    return dest


def refresh() -> None:
    r = subprocess.run(
        ["uv", "run", "python", "tools/collect_experiments.py"], cwd=ROOT, capture_output=True, text=True
    )
    log("experiments refreshed" if r.returncode == 0 else f"collect failed: {r.stderr[-300:]}")


def fold0_scores() -> dict[str, float]:
    """Fold-0 macro F1 (tuned offsets) of every finished full-split run, by run name."""
    scores = {}
    for meta_path in (ROOT / "outputs").rglob("artifacts/*/*/meta.json"):
        meta = json.loads(meta_path.read_text())
        if meta.get("holdout") or meta.get("split_fp") != "10a79e50dde8":
            continue
        s = meta.get("scores", {}).get("macro_f1_tuned")
        if s is not None:
            scores[meta["name"]] = max(s, scores.get(meta["name"], 0))
    return scores


def make_variants(base: str, which: list[str]) -> list[str]:
    cfg = yaml.safe_load((ROOT / "configs" / f"{base}.yaml").read_text())
    SWEEP.mkdir(parents=True, exist_ok=True)
    edits = {
        "ep": {"epochs": cfg.get("epochs", 3) + 2},
        "lr": {"lr": float(cfg.get("lr", 2e-5)) * 2},
        "lang": {"prepend_language": True},
        "cw0": {"class_weight": "none"},
        # bench-a showed mmBERT memorising the train set by epoch 2 (probe loss ~0) while val loss stays flat
        # from epoch 1.25 -> regularise instead of training longer
        "ls": {"label_smoothing": 0.1},
        "ep2": {"epochs": 2},
        # extra ensemble members: same split (split_seed untouched), different training seed
        "s2": {"seed": 7},
        "s3": {"seed": 13},
    }
    names = []
    for w in which:
        name = f"{base}-{w}"
        (SWEEP / f"{name}.yaml").write_text(
            f"# auto-generated variant of configs/{base}.yaml: {edits[w]}\n"
            + yaml.safe_dump({**cfg, **edits[w], "name": name}, sort_keys=False)
        )
        names.append(name)
    return names


def wait_and_fetch(tag: str, kernel: str) -> None:
    """Resume: a stage that is already on Kaggle — wait for it, then download it like run_stage does."""
    unknown_since = None
    while True:
        s = status(kernel)
        if s in ("complete", "error", "cancelacknowledged", "cancelled"):
            break
        if s == "unknown":  # transient API failure: only give up after 15 minutes of it
            unknown_since = unknown_since or time.time()
            if time.time() - unknown_since > 900:
                log(f"[{tag}] status unknown for 15 min — giving up")
                break
        else:
            unknown_since = None
        time.sleep(60)
    dest = ROOT / "outputs" / kernel.split("/")[1]
    dest.mkdir(parents=True, exist_ok=True)
    r = kaggle("kernels", "output", kernel, "-p", str(dest))
    log(f"[{tag}] complete; output -> outputs/{dest.name} ({'ok' if r.returncode == 0 else r.stderr[-200:]})")
    refresh()


def build_v1(name: str) -> None:
    """Merge every full-split run of `name` (fold 0 + folds 1-4) into experiments/07-v1/<name>/."""
    from rmitguard.ensemble import ensemble

    dirs = []
    for meta_path in (ROOT / "outputs").rglob(f"artifacts/*/{name}/meta.json"):
        meta = json.loads(meta_path.read_text())
        if meta.get("split_fp") == "10a79e50dde8" and not meta.get("holdout"):
            dirs.append(meta_path.parent)
    try:
        out = ensemble(dirs, ROOT / "experiments" / "07-v1" / name)
        meta = json.loads((out / "meta.json").read_text())
        s = meta["scores"]
        log(
            f"v1 ready: {name} folds {meta['members'][name]['folds_done']} · OOF macro F1 cross-fit "
            f"{s.get('macro_f1_tuned_crossfit', float('nan')):.4f} · experiments/07-v1/{name}/submission.csv"
        )
    except Exception as exc:
        log(f"v1 ensemble failed: {exc}")


def main() -> None:
    resume = "--resume-after-bench-b" in sys.argv
    if resume:
        log("resumed: waiting for bench-b already on Kaggle")
        wait_and_fetch("bench-b", "longquanton/rmit-sgs-gendata-bench-b-c1")
    else:
        log("overnight start — waiting for sessions already on Kaggle: " + ", ".join(WAIT_FOR))
        for k in WAIT_FOR:
            while status(k) not in ("complete", "error", "cancelacknowledged", "cancelled", "unknown"):
                time.sleep(60)
        log("GPU session free")
        run_stage("bench-a", [cfg_path(n) for n in ENCODERS_A])
        run_stage("bench-b", [cfg_path(n) for n in ENCODERS_B])

    scores = fold0_scores()
    ranked = [n for n in sorted(scores, key=scores.get, reverse=True) if n in ENCODERS_A + ENCODERS_B]
    log("encoder ranking (fold 0, tuned): " + ", ".join(f"{n}={scores[n]:.4f}" for n in ranked))
    if ranked:
        variants = make_variants(ranked[0], ["ls", "ep2", "lang", "cw0"])
        if len(ranked) > 1:
            variants += make_variants(ranked[1], ["ls", "lang"])
        run_stage("tune", [cfg_path(v) for v in variants])

    # v1 = best encoder config on all 5 folds of the full split (fold 0 already exists from bench/tune)
    scores = fold0_scores()
    enc = {n: s for n, s in scores.items() if not n.startswith(("granite", "mistral"))}
    best = max(enc, key=enc.get) if enc else None
    if best:
        log(f"v1 = {best} (fold 0 {enc[best]:.4f}): training folds 1-4")
        STATE.write_text(json.dumps({"best_encoder": best, "scores": scores}, indent=2))
        run_stage("full", [f"{cfg_path(best)}@1,2", f"{cfg_path(best)}@3,4"])
        build_v1(best)
        log(f"holdout for best encoder config: {best}")
        run_stage("holdout", [f"{cfg_path(best)}@0,1,2", f"{cfg_path(best)}@3,4"], holdout=0.2)

    run_stage("llm", [cfg_path(n) for n in LLMS])
    log("overnight finished")


if __name__ == "__main__":
    print(f"started {datetime.now():%Y-%m-%d %H:%M}", flush=True)
    main()
