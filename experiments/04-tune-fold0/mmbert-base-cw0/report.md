# mmbert-base-cw0

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [0] / 5 · 11.37 min · updated 2026-10-22T00:34:07+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9616 |
| macro F1 (tuned offsets, in-sample) | 0.9628 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9527 – 0.9726 |
| offsets (log-prob) | [0.0, 2.2, 0.0] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.975  |   0.966  | 0.9705 |      1736 |
| JAILBREAK        |      0.9381 |   0.9597 | 0.9487 |       868 |
| PROMPT_INJECTION |      0.9792 |   0.9592 | 0.9691 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.9582 |      0.9857 |         0.9943 |                0.8947 |
| fr         |  700 |     0.9704 |      0.9862 |         0.9861 |                0.9388 |
| it         |  559 |     0.9831 |      0.9899 |         0.9857 |                0.9737 |
| pt         | 1260 |     0.9507 |      0.9495 |         0.9025 |                1      |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1677 |               55 |                       4 |
| true_JAILBREAK        |            35 |              833 |                       0 |
| true_PROMPT_INJECTION |             8 |                0 |                     188 |

## Test prediction distribution
- BENIGN: 61.9%
- JAILBREAK: 31.4%
- PROMPT_INJECTION: 6.7%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9616 |   11.1765 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.344 |      0.0187 |    2100 |         1 |         7.6197 |
