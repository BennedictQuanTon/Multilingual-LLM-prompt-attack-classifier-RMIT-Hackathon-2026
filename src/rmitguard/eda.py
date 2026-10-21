"""Quick EDA that answers the questions that drive modelling decisions (writes a Markdown report)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_competition, text_hash


def eda_report(data_dir: str | None = None, out: str | Path = "reports/eda.md") -> str:
    data = load_competition(data_dir)
    train, test = data.train.copy(), data.test.copy()
    for df in (train, test):
        df["n_chars"] = df["text"].str.len()
        df["n_words"] = df["text"].str.split().str.len()
        df["hash"] = df["text"].map(text_hash)

    dup = train.groupby("hash")["label"].agg(["count", "nunique"])
    test_in_train = test["hash"].isin(set(train["hash"]))
    leaked = test[test_in_train].merge(train[["hash", "label"]].drop_duplicates("hash"), on="hash")
    q = [0.5, 0.9, 0.99]
    lines = [
        "# EDA — RMIT Hackathon 2026 (Challenge 1)",
        "",
        f"train {train.shape} · test {test.shape}",
        "",
        "## Label × language (train, counts)",
        pd.crosstab(train["language"], train["label"], margins=True).to_markdown(),
        "",
        "## Label share per language (row %)",
        (pd.crosstab(train["language"], train["label"], normalize="index") * 100).round(1).to_markdown(),
        "",
        "## Language share: train vs test (%)",
        pd.DataFrame(
            {
                "train": train["language"].value_counts(normalize=True) * 100,
                "test": test["language"].value_counts(normalize=True) * 100,
            }
        )
        .round(1)
        .to_markdown(),
        "",
        "## Length in characters (quantiles 50/90/99) — decides max_len",
        train.groupby("label")["n_chars"].quantile(q).unstack().round(0).to_markdown(),
        "",
        f"test: {test['n_chars'].quantile(q).round(0).to_dict()}",
        "",
        "## Duplicates (after lower-case + whitespace normalisation)",
        f"- duplicated groups in train: {(dup['count'] > 1).sum()} "
        f"(rows involved: {int(dup.loc[dup['count'] > 1, 'count'].sum())})",
        f"- groups with CONFLICTING labels: {(dup['nunique'] > 1).sum()}",
        f"- test prompts identical to a train prompt: {int(test_in_train.sum())} "
        f"({test_in_train.mean():.1%}) → label distribution of those: "
        f"{leaked['label'].value_counts().to_dict()}",
        "",
        "## Random examples per label",
    ]
    rng = np.random.default_rng(0)
    for label, g in train.groupby("label"):
        lines.append(f"### {label}")
        for _, row in g.iloc[rng.choice(len(g), min(4, len(g)), replace=False)].iterrows():
            snippet = row["text"][:300].replace("\n", " ").replace("|", "/")
            lines.append(f"- `{row['language']}` {snippet}{'…' if len(row['text']) > 300 else ''}")
        lines.append("")
    text = "\n".join(lines) + "\n"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    return text
