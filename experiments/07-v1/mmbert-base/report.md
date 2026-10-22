# ensemble: mmbert-base

- `mmbert-base` · weight 1.0 · folds [0, 1, 2, 3, 4] · mmbert-base

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 100.0% |
| macro F1 (argmax) | 0.9661 |
| macro F1 (tuned offsets, in-sample) | 0.9667 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9666** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9556 – 0.9761 |
| offsets (log-prob) | [0.0, -0.65, 0.0] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9737 |   0.9805 | 0.9771 |      8680 |
| JAILBREAK        |      0.9645 |   0.9569 | 0.9607 |      4340 |
| PROMPT_INJECTION |      0.9759 |   0.949  | 0.9622 |       980 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         | 1400 |     0.9547 |      0.9891 |         0.992  |                0.883  |
| fr         | 3500 |     0.9772 |      0.991  |         0.9926 |                0.948  |
| it         | 2800 |     0.9784 |      0.9905 |         0.9913 |                0.9534 |
| pt         | 6300 |     0.9579 |      0.9608 |         0.922  |                0.9909 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          8511 |              147 |                      22 |
| true_JAILBREAK        |           186 |             4153 |                       1 |
| true_PROMPT_INJECTION |            44 |                6 |                     930 |

## Test prediction distribution
- BENIGN: 62.3%
- JAILBREAK: 30.9%
- PROMPT_INJECTION: 6.8%

