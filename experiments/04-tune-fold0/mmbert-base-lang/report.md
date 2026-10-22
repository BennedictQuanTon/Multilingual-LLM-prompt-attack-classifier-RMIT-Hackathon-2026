# mmbert-base-lang

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [3, 4] / 5 · 24.79 min · updated 2026-10-22T04:17:19+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 40.0% |
| macro F1 (argmax) | 0.9642 |
| macro F1 (tuned offsets, in-sample) | 0.9684 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9640** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9579 – 0.9783 |
| offsets (log-prob) | [0.0, -2.0, 2.35] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9727 |   0.9844 | 0.9785 |      3472 |
| JAILBREAK        |      0.973  |   0.9556 | 0.9643 |      1736 |
| PROMPT_INJECTION |      0.9764 |   0.949  | 0.9625 |       392 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  559 |     0.9568 |      0.9871 |         0.9914 |                0.8919 |
| fr         | 1400 |     0.9758 |      0.9908 |         0.9942 |                0.9424 |
| it         | 1121 |     0.9688 |      0.9857 |         0.9855 |                0.9351 |
| pt         | 2520 |     0.9661 |      0.9667 |         0.9316 |                1      |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          3418 |               45 |                       9 |
| true_JAILBREAK        |            77 |             1659 |                       0 |
| true_PROMPT_INJECTION |            19 |                1 |                     372 |

## Test prediction distribution
- BENIGN: 63.2%
- JAILBREAK: 29.7%
- PROMPT_INJECTION: 7.1%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      3 |     11200 |    2800 |     0.9659 |   12.2218 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       110.999 |      0.0199 |    2100 |         1 |         7.6197 |
|      4 |     11200 |    2800 |     0.9625 |   12.3008 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       111.098 |      0.0197 |    2100 |         1 |         7.6197 |
