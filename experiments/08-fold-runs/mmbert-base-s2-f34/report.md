# mmbert-base-s2

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [3, 4] / 5 · 23.03 min · updated 2026-10-22T00:11:08+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 40.0% |
| macro F1 (argmax) | 0.9661 |
| macro F1 (tuned offsets, in-sample) | 0.9686 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9620** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9590 – 0.9776 |
| offsets (log-prob) | [0.0, -0.65, 0.9] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9756 |   0.979  | 0.9773 |      3472 |
| JAILBREAK        |      0.9625 |   0.9614 | 0.962  |      1736 |
| PROMPT_INJECTION |      0.9791 |   0.9541 | 0.9664 |       392 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  559 |     0.9739 |      0.9914 |         0.9971 |                0.9333 |
| fr         | 1400 |     0.9752 |      0.9902 |         0.9931 |                0.9424 |
| it         | 1121 |     0.9777 |      0.9892 |         0.9884 |                0.9554 |
| pt         | 2520 |     0.9594 |      0.9617 |         0.925  |                0.9915 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          3399 |               65 |                       8 |
| true_JAILBREAK        |            67 |             1669 |                       0 |
| true_PROMPT_INJECTION |            18 |                0 |                     374 |

## Test prediction distribution
- BENIGN: 62.7%
- JAILBREAK: 30.3%
- PROMPT_INJECTION: 7.0%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      3 |     11200 |    2800 |     0.9662 |   11.3746 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       108.058 |      0.0197 |    2100 |         1 |         7.6197 |
|      4 |     11200 |    2800 |     0.966  |   11.221  | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       108.157 |      0.0196 |    2100 |         1 |         7.6197 |
