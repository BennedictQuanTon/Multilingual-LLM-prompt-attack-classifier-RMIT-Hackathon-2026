"""Cross-lingual near-duplicates: the same source prompt appears translated into several languages.

Embeds every prompt with the *pre-trained* mmBERT-base (an approved backbone, no fine-tuning), removes each
language's mean vector, and finds for every row its most similar train row written in another language.
When that similarity is very high the two prompts are translations of each other and share the label
(99.5% of train pairs at similarity >= 0.80), so the neighbour's label overrides the model's prediction.

    python -m rmitguard xling --out artifacts/xling      # writes xling.npz (≈3 min on a T4)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import APPROVED_MODELS

LAYER = 11  # middle layer: more language-neutral than the last one
# 5-fold OOF (14k rows, mmbert-base-s2): 0.85 fixes 5 errors and breaks none; 0.78 (picked on fold 0 alone)
# fixes 10 but also breaks 10
THRESHOLD = 0.85


def embed(
    texts: list[str], model_path: str, layer: int = LAYER, max_len: int = 256, batch: int = 64
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, output_hidden_states=True).to(device).eval()
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            enc = tok(
                [texts[i] for i in idx],
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).hidden_states[layer]
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            out[idx] = ((hidden * mask).sum(1) / mask.sum(1)).float().cpu().numpy()
    return out


def centre_by_language(emb: np.ndarray, language: np.ndarray) -> np.ndarray:
    emb = emb.copy()
    for lang in np.unique(language):
        m = language == lang
        emb[m] -= emb[m].mean(0)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def nearest_other_language(
    query: np.ndarray,
    q_lang: np.ndarray,
    ref: np.ndarray,
    r_lang: np.ndarray,
    allowed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Index into `ref` and cosine similarity of the closest row in another language.

    `allowed` (n_query × n_ref bool, optional) removes candidates, e.g. rows of the same CV fold.
    """
    sim = query @ ref.T
    sim[q_lang[:, None] == r_lang[None, :]] = -1.0
    if allowed is not None:
        sim[~allowed] = -1.0
    return sim.argmax(1), sim.max(1)


def model_path(key: str) -> str:
    """Kaggle input mount if attached, otherwise the Hugging Face id (same rule as the trainer)."""
    import glob
    import os

    hf_id, handle, _ = APPROVED_MODELS[key]
    slug = handle.split("/")[1]
    hits = [
        os.path.dirname(p)
        for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
        if f"/{slug}/" in p.lower()
    ]
    return sorted(hits, key=len)[0] if hits else hf_id


def run_xling(data_dir: str | None, out_dir: str | Path, n_folds: int = 5, split_seed: int = 42) -> Path:
    from .data import load_competition, make_split

    data = load_competition(data_dir)
    train, test = data.train, data.test
    texts = train["text"].tolist() + test["text"].tolist()
    lang = np.concatenate([train["language"].to_numpy(dtype=str), test["language"].to_numpy(dtype=str)])
    path = model_path("mmbert-base")
    emb = centre_by_language(embed(texts, path), lang)
    e_tr, e_te = emb[: len(train)], emb[len(train) :]
    l_tr, l_te = lang[: len(train)], lang[len(train) :]
    folds = make_split(train, n_folds, split_seed)
    # train rows: neighbours only from OTHER folds -> usable on out-of-fold predictions without leakage
    tr_nn, tr_sim = nearest_other_language(e_tr, l_tr, e_tr, l_tr, allowed=folds[:, None] != folds[None, :])
    te_nn, te_sim = nearest_other_language(e_te, l_te, e_tr, l_tr)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "xling.npz",
        train_nn=tr_nn,
        train_sim=tr_sim,
        test_nn=te_nn,
        test_sim=te_sim,
        folds=folds,
        model=str(APPROVED_MODELS["mmbert-base"][0]),
        layer=LAYER,
    )
    print(
        f"xling: test rows with a translation (sim >= {THRESHOLD}): {(te_sim >= THRESHOLD).sum()} / {len(test)}"
    )
    return out / "xling.npz"


def apply_override(
    probs: np.ndarray, sim: np.ndarray, nn_label: np.ndarray, threshold: float = THRESHOLD
) -> np.ndarray:
    """Replace the probabilities of rows with a near-certain translation by the neighbour's one-hot label."""
    out = probs.copy()
    m = sim >= threshold
    out[m] = np.eye(probs.shape[1], dtype=probs.dtype)[nn_label[m]]
    return out
