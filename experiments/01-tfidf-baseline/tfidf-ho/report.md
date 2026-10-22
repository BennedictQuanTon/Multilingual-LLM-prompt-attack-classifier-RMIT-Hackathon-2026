# tfidf-ho

- model: `tfidf` (local diagnostic, not an approved backbone) · split `c03bfc97563a`
- folds done: [0, 1, 2, 3, 4] / 5 · 1.28 min · updated 2026-10-21T14:36:25+00:00

## Holdout (2802 rows, never trained on, offsets from dev)
| metric | value |
|---|---|
| **holdout macro F1 ← benchmark score** | **0.9253** (95% CI 0.9116 – 0.9372) |
| holdout macro F1 (argmax, no offsets) | 0.9217 |

|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9461 |   0.9499 | 0.948  |      1736 |
| JAILBREAK        |      0.9101 |   0.9206 | 0.9153 |       869 |
| PROMPT_INJECTION |      0.9556 |   0.8731 | 0.9125 |       197 |

| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.9496 |      0.9828 |         0.9943 |                0.8718 |
| fr         |  700 |     0.9347 |      0.9713 |         0.9725 |                0.8602 |
| it         |  560 |     0.9403 |      0.9771 |         0.9829 |                0.8611 |
| pt         | 1261 |     0.9071 |      0.9142 |         0.8361 |                0.9711 |

|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1649 |               79 |                       8 |
| true_JAILBREAK        |            69 |              800 |                       0 |
| true_PROMPT_INJECTION |            25 |                0 |                     172 |

## Dev out-of-fold
| metric | value |
|---|---|
| OOF coverage | 100.0% |
| macro F1 (argmax) | 0.9239 |
| macro F1 (tuned offsets, in-sample) | 0.9283 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9257** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9120 – 0.9423 |
| offsets (log-prob) | [0.0, -0.25, 0.2] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.952  |   0.9518 | 0.9519 |      6944 |
| JAILBREAK        |      0.9089 |   0.9288 | 0.9188 |      3471 |
| PROMPT_INJECTION |      0.9619 |   0.871  | 0.9142 |       783 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         | 1119 |     0.9215 |      0.977  |         0.9674 |                0.8201 |
| fr         | 2800 |     0.9533 |      0.9817 |         0.9844 |                0.8937 |
| it         | 2240 |     0.9374 |      0.9727 |         0.9692 |                0.8704 |
| pt         | 5039 |     0.9113 |      0.9201 |         0.8503 |                0.9635 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          6609 |              309 |                      26 |
| true_JAILBREAK        |           246 |             3224 |                       1 |
| true_PROMPT_INJECTION |            87 |               14 |                     682 |

## Test prediction distribution
- BENIGN: 62.7%
- JAILBREAK: 31.2%
- PROMPT_INJECTION: 6.1%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes |   n_features |
|-------:|----------:|--------:|-----------:|----------:|-------------:|
|      0 |      8959 |    2239 |     0.9193 |    0.2022 |       214869 |
|      1 |      8956 |    2242 |     0.925  |    0.4179 |       214039 |
|      2 |      8958 |    2240 |     0.9277 |    0.6621 |       214848 |
|      3 |      8960 |    2238 |     0.924  |    0.9001 |       218497 |
|      4 |      8959 |    2239 |     0.9233 |    1.1683 |       212641 |
