# tfidf

- model: `tfidf` (local diagnostic, not an approved backbone) · split `10a79e50dde8`
- folds done: [0, 1, 2, 3, 4] / 5 · 1.24 min · updated 2026-10-21T14:30:21+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 100.0% |
| macro F1 (argmax) | 0.9304 |
| macro F1 (tuned offsets, in-sample) | 0.9329 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9314** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9167 – 0.9472 |
| offsets (log-prob) | [0.0, -0.3, 0.25] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9533 |   0.9565 | 0.9549 |      8680 |
| JAILBREAK        |      0.9201 |   0.9265 | 0.9233 |      4340 |
| PROMPT_INJECTION |      0.9501 |   0.8929 | 0.9206 |       980 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         | 1400 |     0.9315 |      0.9799 |         0.9759 |                0.8387 |
| fr         | 3500 |     0.9555 |      0.9819 |         0.9829 |                0.9017 |
| it         | 2800 |     0.9436 |      0.9776 |         0.9782 |                0.875  |
| pt         | 6300 |     0.9158 |      0.924  |         0.8547 |                0.9687 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          8302 |              337 |                      41 |
| true_JAILBREAK        |           314 |             4021 |                       5 |
| true_PROMPT_INJECTION |            93 |               12 |                     875 |

## Test prediction distribution
- BENIGN: 62.7%
- JAILBREAK: 31.1%
- PROMPT_INJECTION: 6.2%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes |   n_features |
|-------:|----------:|--------:|-----------:|----------:|-------------:|
|      0 |     11200 |    2800 |     0.9354 |    0.2147 |       244993 |
|      1 |     11200 |    2800 |     0.9263 |    0.4202 |       244672 |
|      2 |     11200 |    2800 |     0.9374 |    0.6303 |       246696 |
|      3 |     11200 |    2800 |     0.9197 |    0.8893 |       247705 |
|      4 |     11200 |    2800 |     0.9331 |    1.1639 |       245621 |
