"""Merge run directories: fold subsets run by different teammates, and/or several runs (seeds, configs).

Rules enforced here:
- every run must share the same split fingerprint (same folds -> OOF predictions are comparable);
- the competition allows ONE backbone, so mixing backbones needs an explicit --allow-multi-backbone.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import load_competition
from .postprocess import predict_with_bias
from .runner import _jsonable, predictions_frame, score_holdout, score_oof, write_report
from .submission import make_submission, validate_submission


def find_runs(roots: list[str | Path]) -> list[Path]:
    """Every directory below the given roots that contains meta.json + oof.npy + test.npy."""
    found = []
    for root in roots:
        root = Path(root)
        metas = [root / "meta.json"] if (root / "meta.json").exists() else sorted(root.rglob("meta.json"))
        found += [
            m.parent for m in metas if (m.parent / "oof.npy").exists() and (m.parent / "test.npy").exists()
        ]
    return sorted(set(found))


def load_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    hold = run_dir / "holdout.npy"
    return dict(
        dir=run_dir,
        meta=meta,
        oof=np.load(run_dir / "oof.npy"),
        test=np.load(run_dir / "test.npy"),
        holdout=np.load(hold) if hold.exists() else None,
    )


def merge_fold_subsets(runs: list[dict]) -> dict[str, dict]:
    """Same experiment name, different folds -> one run with the union of folds."""
    by_name = defaultdict(list)
    for r in runs:
        by_name[r["meta"]["name"]].append(r)
    merged = {}
    for name, parts in by_name.items():
        oof = np.full_like(parts[0]["oof"], np.nan)
        test_sum, hold_sum, n_models, folds_done, model = 0.0, 0.0, 0, [], parts[0]["meta"]["model"]
        for p in parts:
            overlap = sorted(set(p["meta"]["folds_done"]) & set(folds_done))
            if overlap:
                print(f"⚠️ {name}: fold(s) {overlap} present twice ({p['dir']}); keeping the first copy")
                continue
            m = ~np.isnan(p["oof"]).any(axis=1)
            oof[m] = p["oof"][m]
            test_sum = test_sum + p["test"] * p["meta"]["n_test_models"]
            if p["holdout"] is not None:
                hold_sum = hold_sum + p["holdout"] * p["meta"]["n_test_models"]
            n_models += p["meta"]["n_test_models"]
            folds_done += p["meta"]["folds_done"]
        merged[name] = dict(
            oof=oof,
            test=test_sum / n_models,
            holdout=None if parts[0]["holdout"] is None else hold_sum / n_models,
            folds_done=sorted(folds_done),
            model=model,
            sources=[str(p["dir"]) for p in parts],
        )
    return merged


def ensemble(
    roots: list[str | Path],
    out_dir: str | Path = "artifacts/ensemble",
    data_dir: str | None = None,
    weights: dict[str, float] | None = None,
    allow_multi_backbone: bool = False,
    only: list[str] | None = None,
    xling: str | Path | None = "auto",
    xling_threshold: float | None = None,
    tuned_offsets: bool = False,
) -> Path:
    runs = [
        r for r in (load_run(d) for d in find_runs(roots)) if "members" not in r["meta"]
    ]  # skip old ensembles
    if only:  # e.g. only the B* runs out of a benchmark notebook that trained several models
        runs = [r for r in runs if r["meta"]["name"] in only]
    if not runs:
        raise FileNotFoundError(f"no run directories under {roots}")
    fps = {r["meta"]["split_fp"] for r in runs}
    if len(fps) > 1:
        raise ValueError(
            f"runs use different CV splits {fps}; they cannot be merged (same seed + n_folds needed)"
        )
    merged = merge_fold_subsets(runs)
    backbones = {m["model"] for m in merged.values()}
    if len(backbones) > 1 and not allow_multi_backbone:
        raise ValueError(
            f"runs use several backbones {backbones}; the rules allow ONE approved model. "
            "Pass --allow-multi-backbone only if the organisers confirm it is allowed."
        )

    data = load_competition(data_dir)
    train, test, y = data.train, data.test, data.y
    if len(train) != len(next(iter(merged.values()))["oof"]):
        raise ValueError("runs were made on a different (sub)set of train.csv (e.g. --limit)")
    folds = np.load(runs[0]["dir"] / "folds.npy")
    w = {name: (weights or {}).get(name, 1.0) for name in merged}
    total = sum(w.values())
    oof = sum(w[n] * m["oof"] for n, m in merged.items()) / total  # NaN wherever any member lacks a fold
    test_p = sum(w[n] * m["test"] for n, m in merged.items()) / total

    language = train["language"].astype(str).to_numpy()
    xling_info = None
    if xling == "auto":  # an xling.npz next to the runs (e.g. from a Kaggle input) is picked up automatically
        hits = [p for r in roots for p in Path(r).rglob("xling.npz")] if roots else []
        xling = hits[0] if hits else None
    if xling:
        from .xling import THRESHOLD, apply_override

        z = np.load(xling)
        thr = xling_threshold if xling_threshold is not None else THRESHOLD
        m = ~np.isnan(oof).any(axis=1)
        before = score_oof(oof, y, folds, language)
        if np.array_equal(z["folds"], folds):  # train neighbours come from other folds -> OOF stays honest
            oof = oof.copy()
            oof[m] = apply_override(oof[m], z["train_sim"][m], y[z["train_nn"][m]], thr)
        test_p = apply_override(test_p, z["test_sim"], y[z["test_nn"]], thr)
        xling_info = dict(
            threshold=thr,
            oof_overridden=int((z["train_sim"][m] >= thr).sum()),
            test_overridden=int((z["test_sim"] >= thr).sum()),
            macro_f1_tuned_before=before.get("macro_f1_tuned"),
            macro_f1_crossfit_before=before.get("macro_f1_tuned_crossfit"),
        )
        print(f"xling override (sim >= {thr}): {xling_info}")
    scores = score_oof(oof, y, folds, language)
    # tuned offsets look better in-sample but lose to plain argmax when fitted out-of-fold
    # (5-fold OOF: argmax 0.9649 vs cross-fit 0.9631; 5-member ensemble fold 0: 0.9679 vs 0.9658)
    bias = np.asarray(scores["bias"]) if tuned_offsets else np.zeros(oof.shape[1])
    hold = folds == -1
    holdout = None
    if hold.any():
        hold_p = sum(w[n] * m["holdout"] for n, m in merged.items()) / total
        holdout = score_holdout(hold_p, y[hold], language[hold], bias)
    sub = make_submission(test["id"], predict_with_bias(test_p, bias))
    problems = validate_submission(sub, test["id"], expected_rows=len(test))
    if problems:
        raise RuntimeError(f"invalid submission: {problems}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oof.npy", oof.astype(np.float32))
    np.save(out / "test.npy", test_p.astype(np.float32))
    np.save(out / "folds.npy", folds)
    if holdout is not None:
        np.save(out / "holdout.npy", hold_p.astype(np.float32))
        predictions_frame(train[hold], hold_p, predict_with_bias(hold_p, bias)).to_csv(
            out / "holdout_predictions.csv", index=False
        )
    m = ~np.isnan(oof).any(axis=1)
    predictions_frame(train[m], oof[m], predict_with_bias(oof[m], bias)).to_csv(
        out / "oof_predictions.csv", index=False
    )
    sub.to_csv(out / "submission.csv", index=False)
    members = {
        n: dict(weight=w[n], folds_done=m["folds_done"], model=m["model"], sources=m["sources"])
        for n, m in merged.items()
    }
    (out / "meta.json").write_text(
        json.dumps(
            dict(
                name=out.name,
                model="+".join(sorted(backbones)),
                members=members,
                split_fp=fps.pop(),
                n_train=len(train),
                n_test=len(test),
                scores=_jsonable(scores),
                holdout=_jsonable(holdout),
                xling=xling_info,
                decision="tuned offsets" if tuned_offsets else "argmax",
                updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_report(
        out,
        f"ensemble: {out.name}",
        [f"`{n}` · weight {m['weight']} · folds {m['folds_done']} · {m['model']}" for n, m in members.items()]
        + ([f"cross-lingual override: {xling_info}"] if xling_info else []),
        scores,
        sub,
        [],
        holdout,
    )
    print((out / "report.md").read_text(encoding="utf-8"))
    return out
