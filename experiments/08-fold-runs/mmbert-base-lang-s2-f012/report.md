# mmbert-base-lang-s2

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [0, 1, 2] / 5 · 36.28 min · updated 2026-10-22T05:49:00+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 60.0% |
| macro F1 (argmax) | 0.9643 |
| macro F1 (tuned offsets, in-sample) | 0.9666 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9657** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9560 – 0.9761 |
| offsets (log-prob) | [0.0, 0.5, -3.0] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9766 |   0.9775 | 0.9771 |      5208 |
| JAILBREAK        |      0.9551 |   0.9635 | 0.9593 |      2604 |
| PROMPT_INJECTION |      0.9875 |   0.9405 | 0.9634 |       588 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  841 |     0.9705 |      0.9943 |         0.9885 |                0.9286 |
| fr         | 2100 |     0.9721 |      0.9896 |         0.9901 |                0.9366 |
| it         | 1679 |     0.9853 |      0.9933 |         0.9933 |                0.9694 |
| pt         | 3780 |     0.9541 |      0.959  |         0.9207 |                0.9828 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          5091 |              110 |                       7 |
| true_JAILBREAK        |            95 |             2509 |                       0 |
| true_PROMPT_INJECTION |            27 |                8 |                     553 |

## Test prediction distribution
- BENIGN: 62.1%
- JAILBREAK: 31.6%
- PROMPT_INJECTION: 6.2%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9645 |   11.9719 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       110.288 |      0.0187 |    2100 |         1 |         7.6197 |
|      1 |     11200 |    2800 |     0.9585 |   11.8594 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       110.752 |      0.0198 |    2100 |         1 |         7.6197 |
|      2 |     11200 |    2800 |     0.9699 |   11.9408 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       110.73  |      0.0197 |    2100 |         1 |         7.6197 |
