# mmbert-base-lang-s2-full

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [0] / 5 · 14.41 min · updated 2026-10-22T09:56:33+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 1.0000 |
| macro F1 (tuned offsets, in-sample) | 1.0000 |
| public-LB-sized noise (95% CI, 1800 rows) | 1.0000 – 1.0000 |
| offsets (log-prob) | [0.0, 0.0, 0.0] |

### Per class
|                  |   precision |   recall |   f1 |   support |
|:-----------------|------------:|---------:|-----:|----------:|
| BENIGN           |           1 |        1 |    1 |      1736 |
| JAILBREAK        |           1 |        1 |    1 |       868 |
| PROMPT_INJECTION |           1 |        1 |    1 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |          1 |           1 |              1 |                     1 |
| fr         |  700 |          1 |           1 |              1 |                     1 |
| it         |  559 |          1 |           1 |              1 |                     1 |
| pt         | 1260 |          1 |           1 |              1 |                     1 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1736 |                0 |                       0 |
| true_JAILBREAK        |             0 |              868 |                       0 |
| true_PROMPT_INJECTION |             0 |                0 |                     196 |

## Test prediction distribution
- BENIGN: 62.3%
- JAILBREAK: 31.0%
- PROMPT_INJECTION: 6.7%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     14000 |    2800 |          1 |   14.2346 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       110.773 |      0.0196 |    2625 |         1 |         7.6197 |
