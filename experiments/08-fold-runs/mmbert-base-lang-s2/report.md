# mmbert-base-lang-s2

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [3, 4] / 5 · 25.94 min · updated 2026-10-22T05:38:39+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 40.0% |
| macro F1 (argmax) | 0.9696 |
| macro F1 (tuned offsets, in-sample) | 0.9704 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9698** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9605 – 0.9798 |
| offsets (log-prob) | [0.0, -1.0, 0.0] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9744 |   0.9868 | 0.9805 |      3472 |
| JAILBREAK        |      0.9771 |   0.9568 | 0.9668 |      1736 |
| PROMPT_INJECTION |      0.974  |   0.9541 | 0.9639 |       392 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  559 |     0.9577 |      0.9928 |         0.9914 |                0.8889 |
| fr         | 1400 |     0.9779 |      0.9925 |         0.9977 |                0.9436 |
| it         | 1121 |     0.9713 |      0.9864 |         0.9856 |                0.9419 |
| pt         | 2520 |     0.9679 |      0.9687 |         0.9351 |                1      |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          3426 |               37 |                       9 |
| true_JAILBREAK        |            74 |             1661 |                       1 |
| true_PROMPT_INJECTION |            16 |                2 |                     374 |

## Test prediction distribution
- BENIGN: 63.3%
- JAILBREAK: 29.9%
- PROMPT_INJECTION: 6.7%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      3 |     11200 |    2800 |     0.9677 |   12.8343 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       110.999 |      0.0199 |    2100 |         1 |         7.6197 |
|      4 |     11200 |    2800 |     0.9716 |   12.7058 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       111.098 |      0.0197 |    2100 |         1 |         7.6197 |
