# ensemble: best-now-5members-xling

- `mmbert-base` · weight 1.0 · folds [0] · mmbert-base
- `mmbert-base-cw0` · weight 1.0 · folds [0] · mmbert-base
- `mmbert-base-lang` · weight 1.0 · folds [0] · mmbert-base
- `mmbert-base-ls` · weight 1.0 · folds [0] · mmbert-base
- `mmbert-base-s2` · weight 1.0 · folds [0, 1, 2, 3, 4] · mmbert-base
- cross-lingual override: {'threshold': 0.78, 'oof_overridden': 285, 'test_overridden': 713, 'macro_f1_tuned_before': 0.9698945155770539, 'macro_f1_crossfit_before': None}

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9708 |
| macro F1 (tuned offsets, in-sample) | 0.9728 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9652 – 0.9812 |
| offsets (log-prob) | [0.0, -0.15, -0.65] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9732 |   0.9821 | 0.9776 |      1736 |
| JAILBREAK        |      0.9661 |   0.9516 | 0.9588 |       868 |
| PROMPT_INJECTION |      0.9896 |   0.9745 | 0.982  |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.9673 |      0.9886 |         0.9943 |                0.9189 |
| fr         |  700 |     0.9828 |      0.9908 |         0.9884 |                0.9691 |
| it         |  559 |     0.9957 |      0.9957 |         0.9914 |                1      |
| pt         | 1260 |     0.9579 |      0.9599 |         0.9195 |                0.9944 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1705 |               29 |                       2 |
| true_JAILBREAK        |            42 |              826 |                       0 |
| true_PROMPT_INJECTION |             5 |                0 |                     191 |

## Test prediction distribution
- BENIGN: 62.9%
- JAILBREAK: 30.4%
- PROMPT_INJECTION: 6.7%

