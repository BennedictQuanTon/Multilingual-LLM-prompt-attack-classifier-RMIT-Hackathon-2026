# ensemble: mmbert-base-s2-5fold

- `mmbert-base-s2` · weight 1.0 · folds [0, 1, 2, 3, 4] · mmbert-base

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 100.0% |
| macro F1 (argmax) | 0.9649 |
| macro F1 (tuned offsets, in-sample) | 0.9657 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9631** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9540 – 0.9755 |
| offsets (log-prob) | [0.0, -0.45, 1.05] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9756 |   0.9752 | 0.9754 |      8680 |
| JAILBREAK        |      0.9559 |   0.9597 | 0.9578 |      4340 |
| PROMPT_INJECTION |      0.971  |   0.9571 | 0.964  |       980 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         | 1400 |     0.9768 |      0.9925 |         0.9954 |                0.9424 |
| fr         | 3500 |     0.9735 |      0.9894 |         0.9913 |                0.94   |
| it         | 2800 |     0.978  |      0.9908 |         0.9913 |                0.9519 |
| pt         | 6300 |     0.9535 |      0.9569 |         0.9161 |                0.9875 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          8465 |              189 |                      26 |
| true_JAILBREAK        |           173 |             4165 |                       2 |
| true_PROMPT_INJECTION |            39 |                3 |                     938 |

## Test prediction distribution
- BENIGN: 62.3%
- JAILBREAK: 30.5%
- PROMPT_INJECTION: 7.1%

