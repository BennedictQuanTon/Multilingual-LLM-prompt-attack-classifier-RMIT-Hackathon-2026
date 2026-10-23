# RMIT Hackathon 2026 · Challenge 1 — Team GenData

[![CI](https://github.com/BennedictQuanTon/Team-GenData-RMIT-Hackathon-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/BennedictQuanTon/Team-GenData-RMIT-Hackathon-2026/actions/workflows/ci.yml)

Classifying multilingual LLM prompts (Portuguese, French, Italian, English) as `BENIGN` / `JAILBREAK` / `PROMPT_INJECTION`
with fine-tuned mmBERT-base and a QLoRA-tuned Granite-3.3-2B.

| | |
|---|---|
| **Private LB (final ranking)** | **0.98241** macro F1 · **6th at RMIT Saigon South · 18th region-wide** |
| Best public LB | 0.97913 |
| Single model (distilled mmBERT-base) | 0.97226 private · 0.96514 public |
| Models | 60+ fine-tuning runs on 9 backbones: 35 mmBERT-base CV models, 8 full-data models, Granite-2B 4-bit LoRA |
| Compute | Kaggle 2× T4 (max 2 sessions per account), 21 Oct 20:00 → 22 Oct 17:45 (UTC+7) |

![Selected submissions: public vs private](docs/figures/public_vs_private.png)

**Main lesson:** submission **10** scored only 0.97530 on the public board but had the **best private score (0.98241)**, above
submission 12c (public 0.97913 → private 0.98111). Cross-validation predicted this: both had about the same out-of-fold score (~0.975),
and 12c's extra step changed only 12 of 6,000 labels, so its +0.0036 on the public board (~4 scored rows) was mostly luck.
**Trust 14,000 labelled rows of CV over 1,800 public rows.**

![Public leaderboard by submission](docs/figures/lb_progress.png)

---

## 1. The competition

**RMIT GenAI & Cybersecurity Hackathon 2026** — *Security × Generative AI × Low-Resource Languages* (22–23 Oct 2026, Australia,
Vietnam, Malaysia). Challenge 1 ran on Kaggle: [rmit-hackathon-2026](https://www.kaggle.com/competitions/rmit-hackathon-2026).

Act as an **AI Protector**: read a prompt sent to an LLM and decide whether it is

| label | meaning | share of train |
|---|---|---|
| `BENIGN` | an ordinary request | 62% |
| `JAILBREAK` | talks the model out of its safety rules (role-play, "DAN", fictional scenarios…) | 31% |
| `PROMPT_INJECTION` | hijacks the model's instructions ("ignore previous instructions", system-prompt leaks…) | 7% |

- **Data:** `train.csv` 14,000 rows, `test.csv` 6,000 rows; `language` ∈ {pt 45%, fr 25%, it 20%, en 10%}, every language with the same
  62/31/7 label split. Texts are 10–18,550 characters. Licence CC BY-NC-SA 4.0 (the data is not in this repo).
- **Metric:** macro F1, the unweighted mean of the per-class F1, so the rare `PROMPT_INJECTION` class counts as much as `BENIGN`.
  Public board = 30% of test, private board (final ranking) = the other 70%.
- **Rules:** fine-tune models from the approved list (XLM-R base/large, mmBERT small/base, multilingual DistilBERT, RemBERT,
  DeBERTa-v3 PI-v2, Wolf Defender, Granite 3.3 2B, Mistral 7B v0.3); 6 submissions per day; notebook named `RMIT_SGS_GenData_C1`.
  The repo keeps both an ensemble path (best score) and a single-model path (`configs/single/`).

---

## 2. Solution

```
                  ┌──────────────────────── 5-fold CV (one split, duplicate texts grouped) ────────────────────────┐
 train.csv ──►    │ 7 × mmBERT-base fine-tunes (base, cw0, lang, ls, seed 7, seed 21, lang+seed 7) → 35 models     │
                  │ TF-IDF + logistic regression (CPU)                                             →  5 models     │
                  │ Granite-3.3-2B, 4-bit + LoRA, folds 0/1/3/4 (fold 2 diverged → dropped)        →  4 models     │
                  └────────────────────────────────────────────────────────────────────────────────────────────────┘
 + 2 × mmBERT-base trained on 100% of train (test predictions only)

 probability = 0.9 · mean(mmBERT) + 0.1 · TF-IDF
             → Portuguese rows: 0.6 · above + 0.4 · Granite
             → xling: rows with a translation in train (cosine ≥ 0.85, pre-trained mmBERT) take that row's label
             → argmax
```

**Encoder recipe:** max_len 512 = first 384 + last 128 tokens · AdamW, lr 3e-5, 3 epochs, 10% warm-up · cross-entropy weighted by
√(1/class frequency) · fp16 · frozen embeddings (~½ of mmBERT/XLM-R parameters) · 4 evaluations per epoch on the held-out fold and on
512 training rows, to catch over-fitting. **LLM recipe:** 4-bit NF4 + LoRA r16/α32 on every linear layer, `paged_adamw_8bit`, lr 1e-4,
1 epoch.

**Every decision was made on out-of-fold (OOF) predictions**, never on the public board: 14,000 labelled rows against ~1,800 public
rows. CV tracked the leaderboard throughout (submission 04: fold 0 0.9708 → public 0.97254; submissions 10–11: OOF ~0.975 → public
0.9753–0.9756), and the private score came out ~0.007 above public for all three selected submissions.

![OOF macro F1 step by step](docs/figures/cv_ladder.png)

---

## 3. Findings

### 3.1 Backbone: mmBERT-base wins clearly

Eight encoders on the same fold 0 (2,800 rows), default settings:

![Backbone benchmark](docs/figures/backbones.png)

mmBERT-base (multilingual ModernBERT) is both the most accurate and fast (~12 min per fold on one T4). XLM-R-large needs a whole GPU
and gradient checkpointing and still loses. No hyper-parameter variant (label smoothing, 2 epochs, a language tag, no class weights)
beats the base model on its own, so the variants became ensemble members instead.

### 3.2 Almost every error is Portuguese, BENIGN ↔ JAILBREAK

![Error rate by language](docs/figures/errors_by_language.png)

- Single model: **5.4%** errors in Portuguese (340 of all 429 errors) against ~1.1% in fr/it/en. 328 of the 340 are `BENIGN` ↔
  `JAILBREAK` confusions on 300–3,000-character texts, made with **high confidence** (median p = 0.99).
- The errors are role-play, fiction and "security expert" prompts that look like WildJailbreak's adversarial harmful vs benign pairs:
  both are written the same way and the label depends on the real intent underneath. That needs semantics, hence the LLM.
- Truncation is not the cause: only 35 Portuguese test rows are longer than 2,000 characters.

### 3.3 Granite-2B: weaker alone, complementary where it matters

![Granite vs mmBERT by language](docs/figures/granite_by_language.png)

Granite is below the ensemble in every language, closest in Portuguese, but it makes **different** errors. Blending 40% Granite into
**Portuguese rows only** lifts OOF from 0.9726 to **0.9759** (11,200 rows); blending it into other languages hurts. Public:
0.97254 → **0.97530**; that submission went on to score **0.98241 private**.

**LLM fine-tuning is unstable:** 3 of 8 Granite runs diverged (fold 2 seed 42: F1 0.37 with NaN probabilities; fold 2 seed 7: 0.63;
fold 1 seed 7: 0.68). Rule: **drop any fold with OOF F1 < 0.9** (`tools/build_final.py` does it automatically).

### 3.4 What helped

| change | evidence |
|---|---|
| **Combining runs of one backbone** (variants + seeds) | OOF (14k rows): single 0.9667 → 5 members 0.9718 → 7 members 0.9724 |
| **argmax instead of tuned class offsets** | offsets look better in-sample but lose out-of-fold: OOF 0.9649 argmax vs 0.9631 cross-fitted |
| **xling** (cross-lingual label lookup) | +0.001 OOF at threshold 0.85 (0.78, picked on fold 0 alone, breaks even on 14k rows: 10 fixed, 10 broken) |
| **Granite on Portuguese rows** | +0.0033 OOF, +0.0028 public |
| Models trained on 100% of train (test only) | +0.0036 public (12c), but private **lower** than submission 10 (0.98111 < 0.98241): not a real gain |

### 3.5 What did not help (all measured out-of-fold)

| tried | result |
|---|---|
| Adding other encoders (wolf, deberta, xlmr, distilbert, mmbert-small) | worse |
| TF-IDF weight above 0.1 once the ensemble is strong | no gain; 0.6 → 0.9637 |
| Per-language offsets / pushing only the PI class (cross-fitted) | 0.9639 < 0.9649 argmax / 0.9741 < 0.9749 argmax |
| Stacking (logistic regression on log-probs, per language) | 0.9737 / 0.9715 < 0.9745 hand-set weights |
| Granite only where the ensemble is unsure / only for the B-vs-J decision | 0.9761 / 0.9756 ≈ 0.9759 |
| Voting across translations inside test (predictions only) | 0.9734 → 0.9736 (noise) |
| Same-language kNN (TF-IDF) | fixes nothing: the model is already right on near-duplicates |
| Geometric mean / majority vote instead of arithmetic mean | 0.9721 / 0.9723 vs 0.9724 |

### 3.6 Engineering lessons

- **Do not trust `Trainer.predict` in transformers 5:** it returned rows out of order (XLM-R-large: F1 0.94 during training, 0.35 OOF).
  We wrote our own inference loop and fail the run when the final score drops more than 0.05 below the in-training score.
- **The split must be identical locally and on Kaggle:** `make_folds` uses `numpy.RandomState` and every run records a split
  fingerprint (`10a79e50dde8`).
- **Kaggle sessions can hang forever** (one stuck job holds the session and loses every job's output): the runner kills any job after
  170 minutes, and live logs are read through the API stream (`kernels_logs_stream`) because `kaggle kernels logs` is empty while running.
- **2 sessions per account, CPU included:** run 2 jobs per session (one per T4), and give teammates ready-made notebooks
  (`kaggle/ready/`) to run on their own accounts.
- Submissions differing by a few dozen labels cannot be told apart on the public board (1,800 rows, ±0.004); choose on 11–14k OOF rows.
  The private board confirmed it: the best public submission (12c) was not the best private one (10).

---

## 4. Reproduce

```bash
make setup                     # uv sync + pre-commit
make data                      # competition data into data/ (needs kaggle.json)
make check                     # ruff + pytest

# train on Kaggle: each notebook runs 2 jobs in parallel (one per T4) and kills hung jobs
uv run python tools/build_kernel.py bench configs/mmbert-base.yaml@0,1,2 configs/mmbert-base.yaml@3,4 --tag my-run
kaggle kernels push -p kaggle/build/rmit-sgs-gendata-my-run-c1 -t 3600
kaggle kernels output longquanton/rmit-sgs-gendata-my-run-c1 -p outputs/rmit-sgs-gendata-my-run-c1

uv run rmitguard xling --out artifacts/xling   # cross-lingual neighbours (pre-trained mmBERT, ~3 min on a T4)
make final                     # blend → submissions/RMIT_SGS_GenData_C1.csv (byte-identical to the 0.97913 public submission)
make experiments figures       # experiments/ + docs/figures/
```

Configs: `configs/*.yaml` (benchmark), `configs/sweep/` (variants, seeds, Granite), `configs/single/` (100%-data and distilled single
models). Ready-to-run notebooks for teammates: `kaggle/ready/` (instructions in the first cell).

---

## 5. Repository layout

```
src/rmitguard/        package: config, data (split), models (transformer / LLM / TF-IDF), runner (CV + artifacts),
                      ensemble (per-language weights), xling, postprocess, charts, cli
tools/                build_kernel (Kaggle notebooks) · build_final (final blend) · select_final (score members out-of-fold)
                      collect_experiments · make_readme_figures · push_when_free / fetch_when_done (Kaggle session queue)
configs/              one file per experiment (the model must be on the approved list; checked on load)
experiments/          every run: training-health curves, confusion, errors by language/length, comparisons (own README)
docs/                 PROBLEM.md (task) · WORKFLOW.md (workflow) · figures/ (README charts)
kaggle/ready/         notebooks for teammates to run on their own accounts
tests/                unit + end-to-end tests on synthetic data
```

Never committed: competition data, `*.npy`, weights, `kaggle.json`, `outputs/`, `artifacts/`, submissions (`.gitignore` + a CI check).

---

## 6. Submissions

| # | change | public | private |
|---|---|---|---|
| 02 | 5 mmBERT-base (fold 0) + tuned offsets | 0.96915 | |
| 04 | + xling, argmax | 0.97254 | |
| distill ✓ | **single mmBERT-base**, distilled from the ensemble, 100% of train | 0.96514 | 0.97226 |
| **10 ✓** | **7 mmBERT × 5 folds + TF-IDF + Granite (Portuguese, 2 folds) + xling** | 0.97530 | **0.98241** |
| 11 | Granite on 4 folds | 0.97555 | |
| 12a | Granite Portuguese weight 0.45 | 0.97599 | |
| 12b | Granite only where the ensemble is unsure | 0.97644 | |
| 12c ✓ | + 2 mmBERT-base trained on 100% of train → `RMIT_SGS_GenData_C1.csv` | **0.97913** | 0.98111 |
| 13 | + 6 full-data models, Granite seed 2 | 0.97824 | |

✓ = the three submissions selected for final scoring (Kaggle ranks on the best of them). Private scores are shown only for selected
submissions.

---

Team **GenData** · RMIT Saigon South · RMIT GenAI & Cybersecurity Hackathon 2026.
The team's Challenge 2 (Tide UNLOCKED): [RMIT_SGS_GenData_C2](https://github.com/BennedictQuanTon/RMIT_SGS_GenData_C2) — *Unfound*.
