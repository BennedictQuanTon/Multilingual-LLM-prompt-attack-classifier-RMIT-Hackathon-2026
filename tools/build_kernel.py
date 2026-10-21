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
from rmitguard.constants import COMPETITION  # noqa: E402

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
            "import sys, importlib\nsys.path.insert(0, 'src')\nimport rmitguard\nimportlib.reload(rmitguard)\n"
            "import torch, transformers\n"
            "print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],\n"
            "      '| torch', torch.__version__, '| transformers', transformers.__version__)"
        )
    )
    return cells


def build(args) -> Path:
    team = yaml.safe_load((ROOT / "kaggle" / "team.yaml").read_text())
    model_sources, kernel_sources, enable_gpu = [], [], True

    if args.mode == "train":
        overrides = {"folds": [int(x) for x in args.folds.split(",")] if args.folds else None}
        cfg = load_config(args.config, **overrides)
        if cfg.kind == "tfidf":
            raise SystemExit(
                "tfidf is a local diagnostic, not an approved backbone; do not push it to Kaggle"
            )
        config_text = (ROOT / args.config).read_text(encoding="utf-8")
        suffix = (
            None
            if args.final
            else cfg.name + (f"-f{''.join(map(str, cfg.fold_ids()))}" if args.folds else "")
        )
        model_sources = [cfg.kaggle_model]
        header = (
            f"**Chosen model (approved list): `{cfg.hf_id}`** — Kaggle model `{cfg.kaggle_model}`.\n\n"
            f"Experiment `{cfg.name}` · folds {cfg.fold_ids()} of {cfg.n_folds} · "
            f"lr {cfg.lr} · epochs {cfg.epochs} · max_len {cfg.max_len} · class weights `{cfg.class_weight}`"
        )
        folds_literal = repr(overrides["folds"])
        run = [
            md("## Configuration"),
            code(
                f'CONFIG_YAML = """\n{config_text.rstrip()}\n"""\nFOLDS = {folds_literal}  # None = all folds\n\n'
                "from rmitguard.config import load_config_text\n"
                "cfg = load_config_text(CONFIG_YAML, folds=FOLDS)\ncfg"
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
    elif args.mode == "ensemble":
        kernel_sources = [s for s in (args.sources or "").split(",") if s]
        if not kernel_sources:
            raise SystemExit("--sources user/kernel-slug[,user/kernel-slug...] is required for an ensemble")
        suffix = None if args.final else "ensemble"
        enable_gpu = False
        header = (
            "Merges the out-of-fold and test predictions of our training notebooks (same CV split, same backbone),\n"
            "re-tunes the macro-F1 decision offsets on the merged OOF and writes `submission.csv`.\n\n"
            "Inputs: " + ", ".join(f"`{s}`" for s in kernel_sources)
        )
        run = [
            md("## Merge runs"),
            code(
                "from rmitguard.ensemble import ensemble\n"
                f"out = ensemble(['/kaggle/input'], out_dir='artifacts/ensemble', allow_multi_backbone={args.allow_multi_backbone})"
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
    parser.add_argument("mode", choices=["train", "ensemble", "eda"])
    parser.add_argument("config", nargs="?", help="experiment YAML (train mode)")
    parser.add_argument("--folds", default=None, help="comma-separated fold subset, e.g. 0,1")
    parser.add_argument("--sources", default=None, help="ensemble: comma-separated kernel refs to merge")
    parser.add_argument("--allow-multi-backbone", action="store_true")
    parser.add_argument("--final", action="store_true", help="use the official name RMIT_SGS_GenData_C1")
    parser.add_argument("--username", default=None)
    args = parser.parse_args(argv)
    if args.mode == "train" and not args.config:
        parser.error("train mode needs a config path")
    build(args)


if __name__ == "__main__":
    main()
