# mmbert-base-distill

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [0] / 5 · 14.12 min · updated 2026-10-22T06:37:10+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9663 |
| macro F1 (tuned offsets, in-sample) | 0.9698 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9598 – 0.9782 |
| offsets (log-prob) | [0.0, -0.45, -1.05] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9726 |   0.9821 | 0.9774 |      1736 |
| JAILBREAK        |      0.9662 |   0.9551 | 0.9606 |       868 |
| PROMPT_INJECTION |      0.9894 |   0.9541 | 0.9714 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.9445 |      0.9799 |         0.9886 |                0.8649 |
| fr         |  700 |     0.9788 |      0.9897 |         0.9884 |                0.9583 |
| it         |  559 |     0.9859 |      0.9928 |         0.9914 |                0.9737 |
| pt         | 1260 |     0.9627 |      0.9631 |         0.9249 |                1      |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1705 |               29 |                       2 |
| true_JAILBREAK        |            39 |              829 |                       0 |
| true_PROMPT_INJECTION |             9 |                0 |                     187 |

## Test prediction distribution
- BENIGN: 62.7%
- JAILBREAK: 30.7%
- PROMPT_INJECTION: 6.6%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9663 |   13.7446 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.344 |      0.0187 |    2100 |         1 |         7.6197 |
