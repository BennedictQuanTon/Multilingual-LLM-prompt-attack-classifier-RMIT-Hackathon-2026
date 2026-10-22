# mmbert-base-s4

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [3, 4] / 5 · 22.88 min · updated 2026-10-22T06:02:00+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 40.0% |
| macro F1 (argmax) | 0.9676 |
| macro F1 (tuned offsets, in-sample) | 0.9679 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9645** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9571 – 0.9768 |
| offsets (log-prob) | [0.0, -2.45, 0.0] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9718 |   0.9839 | 0.9778 |      3472 |
| JAILBREAK        |      0.9719 |   0.9551 | 0.9634 |      1736 |
| PROMPT_INJECTION |      0.9789 |   0.9464 | 0.9624 |       392 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  559 |     0.9627 |      0.987  |         0.9943 |                0.9067 |
| fr         | 1400 |     0.9788 |      0.9925 |         0.9965 |                0.9474 |
| it         | 1121 |     0.9727 |      0.9878 |         0.9884 |                0.9419 |
| pt         | 2520 |     0.9603 |      0.9633 |         0.9262 |                0.9915 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          3416 |               48 |                       8 |
| true_JAILBREAK        |            78 |             1658 |                       0 |
| true_PROMPT_INJECTION |            21 |                0 |                     371 |

## Test prediction distribution
- BENIGN: 63.8%
- JAILBREAK: 29.6%
- PROMPT_INJECTION: 6.6%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      3 |     11200 |    2800 |     0.9696 |   11.243  | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       108.058 |      0.0197 |    2100 |         1 |         7.6197 |
|      4 |     11200 |    2800 |     0.9655 |   11.3362 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       108.157 |      0.0196 |    2100 |         1 |         7.6197 |
