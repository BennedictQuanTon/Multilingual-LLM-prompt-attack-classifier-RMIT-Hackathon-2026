# mmbert-base-lang

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [1, 2] / 5 · 24.72 min · updated 2026-10-22T04:16:54+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 40.0% |
| macro F1 (argmax) | 0.9669 |
| macro F1 (tuned offsets, in-sample) | 0.9687 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9664** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9588 – 0.9779 |
| offsets (log-prob) | [0.0, 0.5, -1.75] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.979  |   0.9784 | 0.9787 |      3472 |
| JAILBREAK        |      0.9578 |   0.9672 | 0.9625 |      1736 |
| PROMPT_INJECTION |      0.9841 |   0.9464 | 0.9649 |       392 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  560 |     0.9781 |      0.9957 |         0.9914 |                0.9474 |
| fr         | 1400 |     0.9763 |      0.9908 |         0.9908 |                0.9474 |
| it         | 1120 |     0.9751 |      0.9914 |         0.9928 |                0.9412 |
| pt         | 2520 |     0.9593 |      0.9624 |         0.927  |                0.9886 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          3397 |               69 |                       6 |
| true_JAILBREAK        |            57 |             1679 |                       0 |
| true_PROMPT_INJECTION |            16 |                5 |                     371 |

## Test prediction distribution
- BENIGN: 61.8%
- JAILBREAK: 31.8%
- PROMPT_INJECTION: 6.4%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      1 |     11200 |    2800 |     0.965  |   12.163  | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       110.752 |      0.0198 |    2100 |         1 |         7.6197 |
|      2 |     11200 |    2800 |     0.9688 |   12.2728 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       110.73  |      0.0197 |    2100 |         1 |         7.6197 |
