# mmbert-base-ls

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [3, 4] / 5 · 24.8 min · updated 2026-10-22T05:06:59+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 40.0% |
| macro F1 (argmax) | 0.9605 |
| macro F1 (tuned offsets, in-sample) | 0.9644 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9627** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9542 – 0.9741 |
| offsets (log-prob) | [0.0, -1.15, -0.65] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9722 |   0.9781 | 0.9752 |      3472 |
| JAILBREAK        |      0.9627 |   0.9528 | 0.9577 |      1736 |
| PROMPT_INJECTION |      0.964  |   0.9566 | 0.9603 |       392 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  559 |     0.9615 |      0.9899 |         0.9971 |                0.8974 |
| fr         | 1400 |     0.9754 |      0.9902 |         0.9931 |                0.943  |
| it         | 1121 |     0.9738 |      0.9871 |         0.9855 |                0.9487 |
| pt         | 2520 |     0.9545 |      0.9583 |         0.9166 |                0.9887 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          3396 |               64 |                      12 |
| true_JAILBREAK        |            80 |             1654 |                       2 |
| true_PROMPT_INJECTION |            17 |                0 |                     375 |

## Test prediction distribution
- BENIGN: 63.0%
- JAILBREAK: 30.2%
- PROMPT_INJECTION: 6.7%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      3 |     11200 |    2800 |     0.9593 |   12.229  | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       108.058 |      0.0197 |    2100 |         1 |         7.6197 |
|      4 |     11200 |    2800 |     0.9617 |   12.2846 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       108.157 |      0.0196 |    2100 |         1 |         7.6197 |
