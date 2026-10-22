# mmbert-base

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [0] / 5 · 12.29 min · updated 2026-10-21T16:11:08+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9622 |
| macro F1 (tuned offsets, in-sample) | 0.9649 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9544 – 0.9746 |
| offsets (log-prob) | [0.0, 2.25, -3.0] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9736 |   0.9758 | 0.9747 |      1736 |
| JAILBREAK        |      0.9552 |   0.9574 | 0.9563 |       868 |
| PROMPT_INJECTION |      0.9789 |   0.949  | 0.9637 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.9438 |      0.9858 |         0.9886 |                0.8571 |
| fr         |  700 |     0.9704 |      0.9862 |         0.9861 |                0.9388 |
| it         |  559 |     0.9859 |      0.9928 |         0.9914 |                0.9737 |
| pt         | 1260 |     0.9563 |      0.9577 |         0.917  |                0.9944 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1694 |               38 |                       4 |
| true_JAILBREAK        |            37 |              831 |                       0 |
| true_PROMPT_INJECTION |             9 |                1 |                     186 |

## Test prediction distribution
- BENIGN: 62.7%
- JAILBREAK: 30.9%
- PROMPT_INJECTION: 6.4%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9622 |   12.1046 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.344 |      0.0187 |    2100 |         1 |         7.6197 |
