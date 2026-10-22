"""Build a self-contained Kaggle notebook + kernel-metadata.json from the repo.

The package source is embedded as `%%writefile` cells, so the notebook runs on Kaggle with no extra dataset,
stays readable for the judges, and is always exactly what is in git.

Usage:
    python tools/build_kernel.py train configs/xlmr-large.yaml [--folds 0,1] [--final]
    python tools/build_kernel.py ensemble --sources user/kernel-a,user/kernel-b [--final]
    python tools/build_kernel.py eda
Then: kaggle kernels push -p kaggle/build/<slug>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "rmitguard"
sys.path.insert(0, str(ROOT / "src"))

from rmitguard.config import load_config  # noqa: E402
from rmitguard.constants import APPROVED_MODELS, COMPETITION  # noqa: E402

NAME_RULE = {"university": r"[A-Za-z]+", "campus": r"[A-Za-z]+", "team": r"[A-Za-z0-9_-]+"}


def kaggle_username(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if cred.exists():
        return json.loads(cred.read_text())["username"]
    raise SystemExit(
        "Kaggle username unknown: pass --username, set KAGGLE_USERNAME or create ~/.kaggle/kaggle.json"
    )


def notebook_title(team: dict, suffix: str | None) -> str:
    for key, rule in NAME_RULE.items():
        if not re.fullmatch(rule, str(team[key])):
            raise SystemExit(f"kaggle/team.yaml: {key}={team[key]!r} breaks the naming rule ({rule})")
    team_part = team["team"] if not suffix else f"{team['team']}-{re.sub(r'[^A-Za-z0-9-]+', '-', suffix)}"
    return f"{team['university']}_{team['campus']}_{team_part}_C{team['challenge']}"


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def md(text: str) -> dict:
    lines = text.strip("\n").splitlines()
    return {"cell_type": "markdown", "metadata": {}, "source": [x + "\n" for x in lines[:-1]] + lines[-1:]}


def code(text: str) -> dict:
    lines = text.strip("\n").splitlines()
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [x + "\n" for x in lines[:-1]] + lines[-1:],
    }


def package_cells() -> list[dict]:
    files = sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)
    dirs = sorted({str(p.parent.relative_to(PKG.parent)) for p in files})
    cells = [
        md(
            "## Source code\nThe `rmitguard` package from our repository, written to disk so the run is reproducible."
        ),
        code("import os\n" + "\n".join(f"os.makedirs('src/{d}', exist_ok=True)" for d in dirs)),
    ]
    for p in files:
        rel = p.relative_to(PKG.parent).as_posix()
        cells.append(code(f"%%writefile src/{rel}\n" + p.read_text(encoding="utf-8").rstrip("\n")))
    cells.append(
        code(
            "import os, sys, importlib\nos.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')\nsys.path.insert(0, 'src')\nimport rmitguard\nimportlib.reload(rmitguard)\n"
            "import torch, transformers\n"
            "print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],\n"
            "      '| torch', torch.__version__, '| transformers', transformers.__version__)"
        )
    )
    return cells


def _folds(value: str | None) -> list[int] | None:
    return [int(x) for x in value.split(",")] if value else None


BENCH_RUNNER = """import os, subprocess, sys, time
from pathlib import Path
import torch

Path('configs_run').mkdir(exist_ok=True)
Path('logs').mkdir(exist_ok=True)
FOLDS = {job: folds for job, _, folds in JOBS}
for job, text, _ in JOBS:
    if text:
        Path(f'configs_run/{job}.yaml').write_text(text)
pending = [job for job, _, _ in JOBS]
gpus = list(range(max(1, torch.cuda.device_count())))
running, status, t0 = {}, {}, time.time()


def launch(gpu, name):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': 'src'}
    cmd = [sys.executable, '-m', 'rmitguard', 'train', f'configs_run/{name}.yaml', '--out', f'artifacts/{name}']
    if name == 'xling':  # cross-lingual neighbours (pre-trained mmBERT-base embeddings), a few minutes on a T4
        cmd = [sys.executable, '-m', 'rmitguard', 'xling', '--out', 'artifacts/xling']
    elif FOLDS[name]:
        cmd += ['--folds', FOLDS[name]]
    if HOLDOUT:
        cmd += ['--holdout', str(HOLDOUT)]
    return subprocess.Popen(cmd, env=env, stdout=open(f'logs/{name}.log', 'w'), stderr=subprocess.STDOUT)


JOB_LIMIT_MIN = 170  # a hung job must not keep the session (and every other job's output) alive forever
started = {}
while pending or running:
    for gpu in gpus:
        if gpu in running and running[gpu][1].poll() is None and time.time() - started[gpu] > JOB_LIMIT_MIN * 60:
            print(f'[{(time.time() - t0) / 60:5.1f} min] {running[gpu][0]} killed: over {JOB_LIMIT_MIN} min', flush=True)
            running[gpu][1].kill()
            running[gpu][1].wait()
        if gpu in running and running[gpu][1].poll() is not None:
            name, proc = running.pop(gpu)
            status[name] = proc.returncode
            print(f'[{(time.time() - t0) / 60:5.1f} min] {name} finished on GPU {gpu}: rc={proc.returncode}', flush=True)
        if gpu not in running and pending:
            name = pending.pop(0)
            running[gpu] = (name, launch(gpu, name))
            started[gpu] = time.time()
            print(f'[{(time.time() - t0) / 60:5.1f} min] {name} started on GPU {gpu}', flush=True)
    time.sleep(20)
status"""

BENCH_SUMMARY = """import json, shutil
from pathlib import Path
for name, rc in status.items():
    print(f'===== {name} (rc={rc})')
    if rc != 0:
        print(''.join(Path(f'logs/{name}.log').read_text().splitlines(True)[-40:]))
        continue
    if name == 'xling':
        print(''.join(Path('logs/xling.log').read_text().splitlines(True)[-3:]))
        continue
    meta = json.loads(next(Path(f'artifacts/{name}').glob('*/meta.json')).read_text())
    s, ho = meta['scores'], meta.get('holdout')
    print('fold macro F1 (argmax):', [round(f['macro_f1'], 4) for f in meta['fold_stats']],
          '| tuned:', round(s['macro_f1_tuned'], 4), '| holdout:', ho and round(ho['macro_f1'], 4),
          '| minutes:', [round(f['minutes'], 1) for f in meta['fold_stats']])
ok = [n for n, rc in status.items() if rc == 0]
if ok:
    shutil.copy(next(Path(f'artifacts/{ok[0]}').glob('*/submission.csv')), 'submission.csv')"""


def build(args) -> Path:
    team = yaml.safe_load((ROOT / "kaggle" / "team.yaml").read_text())
    model_sources, kernel_sources, enable_gpu, setup = [], [], True, []

    if args.mode == "train":
        overrides = {
            "folds": [int(x) for x in args.folds.split(",")] if args.folds else None,
            "holdout": args.holdout,
        }
        cfg = load_config(args.config[0], **overrides)
        if cfg.kind == "tfidf":
            raise SystemExit(
                "tfidf is a local diagnostic, not an approved backbone; do not push it to Kaggle"
            )
        config_text = (ROOT / args.config[0]).read_text(encoding="utf-8")
        suffix = (
            None
            if args.final
            else cfg.name
            + ("-ho" if cfg.holdout else "")
            + (f"-f{''.join(map(str, cfg.fold_ids()))}" if args.folds else "")
        )
        model_sources = [cfg.kaggle_model]
        if cfg.kind == "llm":  # 4-bit loading + LoRA are not guaranteed in the Kaggle image
            setup = [code("!pip install -q -U bitsandbytes peft")]
        header = (
            f"**Chosen model (approved list): `{cfg.hf_id}`** — Kaggle model `{cfg.kaggle_model}`.\n\n"
            f"Experiment `{cfg.name}` · folds {cfg.fold_ids()} of {cfg.n_folds} · "
            + (f"**holdout {cfg.holdout:.0%} locked away (private-LB simulation)** · " if cfg.holdout else "")
            + f"lr {cfg.lr} · epochs {cfg.epochs} · max_len {cfg.max_len} · class weights `{cfg.class_weight}`"
        )
        folds_literal = repr(overrides["folds"])
        holdout_literal = repr(args.holdout)
        run = [
            md("## Configuration"),
            code(
                f'CONFIG_YAML = """\n{config_text.rstrip()}\n"""\nFOLDS = {folds_literal}  # None = all folds\n'
                f"HOLDOUT = {holdout_literal}  # None = as in the config\n\n"
                "from rmitguard.config import load_config_text\n"
                "cfg = load_config_text(CONFIG_YAML, folds=FOLDS, holdout=HOLDOUT)\ncfg"
            ),
            md("## Train with K-fold CV\nArtifacts are saved after every fold under `artifacts/<name>/`."),
            code(
                "from rmitguard.runner import run_experiment\nrun_dir = run_experiment(cfg, out_dir='artifacts')"
            ),
            code(
                "import shutil\nshutil.copy(run_dir / 'submission.csv', 'submission.csv')\n"
                "from IPython.display import Markdown\nMarkdown((run_dir / 'report.md').read_text())"
            ),
        ]
    elif args.mode == "bench":
        # several configs in ONE session: each T4 trains its own model (a Kaggle account gets one GPU
        # session at a time, so this doubles throughput and halves the quota spent)
        with_xling = "xling" in args.config
        specs = [
            (c.split("@")[0], c.split("@")[1] if "@" in c else args.folds)
            for c in args.config
            if c != "xling"
        ]
        cfgs = [load_config(p, folds=_folds(f), holdout=args.holdout) for p, f in specs]
        if any(c.kind == "tfidf" for c in cfgs):
            raise SystemExit("tfidf is a local diagnostic; do not push it to Kaggle")
        suffix = args.tag or "bench"
        model_sources = sorted({c.kaggle_model for c in cfgs})
        if any(c.kind == "llm" for c in cfgs):
            setup = [code("!pip install -q -U bitsandbytes peft")]
        header = "Benchmark: one model per GPU, in parallel.\n\n" + "\n".join(
            f"- `{c.name}`: **{c.hf_id}** · folds {c.fold_ids()} · lr {c.lr} · epochs {c.epochs} · max_len {c.max_len}"
            + (f" · holdout {c.holdout:.0%}" if c.holdout else "")
            for c in cfgs
        )
        jobs = [
            (
                f"{c.name}" + (f"-f{f.replace(',', '')}" if f else ""),
                (ROOT / p).read_text(encoding="utf-8"),
                f,
            )
            for c, (p, f) in zip(cfgs, specs)
        ] + ([("xling", "", None)] if with_xling else [])
        if with_xling and APPROVED_MODELS["mmbert-base"][1] not in model_sources:
            model_sources.append(APPROVED_MODELS["mmbert-base"][1])
        run = [
            md("## Run every config on its own GPU"),
            code(
                f"JOBS = {jobs!r}  # (job id, config yaml, folds)\nHOLDOUT = {args.holdout!r}\n\n"
                + BENCH_RUNNER
            ),
            code(BENCH_SUMMARY),
        ]
    elif args.mode == "ensemble":
        kernel_sources = [s for s in (args.sources or "").split(",") if s]
        if not kernel_sources:
            raise SystemExit("--sources user/kernel-slug[,user/kernel-slug...] is required for an ensemble")
        suffix = None if args.final else "ensemble"
        enable_gpu = False
        hf_id, kaggle_model, _ = APPROVED_MODELS[args.backbone]
        header = (
            f"**Chosen model (approved list): `{hf_id}`** — Kaggle model `{kaggle_model}`. "
            "Every ensemble member is a fine-tuned copy of this one backbone (different seeds / training options).\n\n"
            "Merges the out-of-fold and test predictions of our training notebooks (same CV split, same backbone),\n"
            "re-tunes the macro-F1 decision offsets on the merged OOF and writes `submission.csv`.\n\n"
            "Inputs: " + ", ".join(f"`{s}`" for s in kernel_sources)
        )
        run = [
            md("## Merge runs"),
            code(
                "from rmitguard.ensemble import ensemble\n"
                f"out = ensemble(['/kaggle/input'], out_dir='artifacts/ensemble', allow_multi_backbone={args.allow_multi_backbone}, only={args.only.split(',') if args.only else None!r})"
            ),
            code(
                "import shutil\nshutil.copy(out / 'submission.csv', 'submission.csv')\n"
                "from IPython.display import Markdown\nMarkdown((out / 'report.md').read_text())"
            ),
        ]
    else:  # eda
        suffix, enable_gpu = "eda", False
        header = "Exploratory data analysis of the competition data."
        run = [
            code(
                "from rmitguard.eda import eda_report\nfrom IPython.display import Markdown\n"
                "Markdown(eda_report(out='eda.md'))"
            )
        ]

    title = notebook_title(team, suffix)
    slug = slugify(title)
    cells = [
        md(
            f"# {title}\n\nRMIT Hackathon 2026 · Challenge 1 — multilingual prompt-attack classification "
            f"(BENIGN / JAILBREAK / PROMPT_INJECTION), metric: macro F1.\n\n{header}"
        ),
        *setup,
        *package_cells(),
        *run,
    ]
    nb = {
        "cells": cells,
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    }
    out = ROOT / "kaggle" / "build" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    meta = {
        "id": f"{kaggle_username(args.username)}/{slug}",
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [COMPETITION],
        "kernel_sources": kernel_sources,
        "model_sources": model_sources,
    }
    if enable_gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {out.relative_to(ROOT)}  ({meta['id']}, {len(cells)} cells)")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["train", "bench", "ensemble", "eda"])
    parser.add_argument(
        "config", nargs="*", help="experiment YAML (train: one; bench: several, one GPU each)"
    )
    parser.add_argument("--folds", default=None, help="comma-separated fold subset, e.g. 0,1")
    parser.add_argument("--holdout", type=float, default=None, help="lock this share of train away, e.g. 0.2")
    parser.add_argument("--sources", default=None, help="ensemble: comma-separated kernel refs to merge")
    parser.add_argument("--allow-multi-backbone", action="store_true")
    parser.add_argument("--only", default=None, help="ensemble: comma-separated run names to merge")
    parser.add_argument(
        "--backbone", default="mmbert-base", help="ensemble: approved model key stated at the top"
    )
    parser.add_argument("--final", action="store_true", help="use the official name RMIT_SGS_GenData_C1")
    parser.add_argument("--username", default=None)
    parser.add_argument("--tag", default=None, help="bench: name suffix of the notebook")
    args = parser.parse_args(argv)
    if args.mode == "train" and len(args.config) != 1:
        parser.error("train mode needs exactly one config path (use bench for several)")
    if args.mode == "bench" and not args.config:
        parser.error("bench mode needs config paths")
    build(args)


if __name__ == "__main__":
    main()
