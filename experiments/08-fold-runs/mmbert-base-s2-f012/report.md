# mmbert-base-s2

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [0, 1, 2] / 5 · 33.84 min · updated 2026-10-22T00:21:56+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 60.0% |
| macro F1 (argmax) | 0.9642 |
| macro F1 (tuned offsets, in-sample) | 0.9651 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9634** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9544 – 0.9757 |
| offsets (log-prob) | [0.0, -0.45, -2.65] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9739 |   0.9748 | 0.9744 |      5208 |
| JAILBREAK        |      0.9519 |   0.9578 | 0.9548 |      2604 |
| PROMPT_INJECTION |      0.9841 |   0.949  | 0.9662 |       588 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  841 |     0.9715 |      0.9924 |         0.9923 |                0.9298 |
| fr         | 2100 |     0.9759 |      0.99   |         0.99   |                0.9477 |
| it         | 1679 |     0.9854 |      0.9933 |         0.9933 |                0.9697 |
| pt         | 3780 |     0.9487 |      0.9532 |         0.9101 |                0.9828 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          5077 |              122 |                       9 |
| true_JAILBREAK        |           110 |             2494 |                       0 |
| true_PROMPT_INJECTION |            26 |                4 |                     558 |

## Test prediction distribution
- BENIGN: 62.9%
- JAILBREAK: 30.8%
- PROMPT_INJECTION: 6.3%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9635 |   11.1995 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.344 |      0.0187 |    2100 |         1 |         7.6197 |
|      1 |     11200 |    2800 |     0.96   |   11.0574 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.811 |      0.0196 |    2100 |         1 |         7.6197 |
|      2 |     11200 |    2800 |     0.9689 |   11.0293 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.789 |      0.0196 |    2100 |         1 |         7.6197 |
