# tfidf-f0

- model: `tfidf` (local diagnostic, not an approved backbone) · split `10a79e50dde8`
- folds done: [0] / 5 · 0.23 min · updated 2026-10-21T14:34:02+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9354 |
| macro F1 (tuned offsets, in-sample) | 0.9378 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9213 – 0.9509 |
| offsets (log-prob) | [0.0, -0.2, -0.25] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.959  |   0.9562 | 0.9576 |      1736 |
| JAILBREAK        |      0.9176 |   0.9366 | 0.927  |       868 |
| PROMPT_INJECTION |      0.9617 |   0.898  | 0.9288 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.9115 |      0.9799 |         0.9667 |                0.7879 |
| fr         |  700 |     0.953  |      0.9748 |         0.9676 |                0.9167 |
| it         |  559 |     0.949  |      0.9812 |         0.9828 |                0.8831 |
| pt         | 1260 |     0.9289 |      0.9324 |         0.8715 |                0.9827 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1660 |               69 |                       7 |
| true_JAILBREAK        |            55 |              813 |                       0 |
| true_PROMPT_INJECTION |            16 |                4 |                     176 |

## Test prediction distribution
- BENIGN: 62.5%
- JAILBREAK: 31.4%
- PROMPT_INJECTION: 6.0%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes |   n_features |
|-------:|----------:|--------:|-----------:|----------:|-------------:|
|      0 |     11200 |    2800 |     0.9354 |    0.2162 |       244993 |
