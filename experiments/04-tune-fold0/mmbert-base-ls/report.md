# mmbert-base-ls

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [0] / 5 · 11.22 min · updated 2026-10-22T00:22:38+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9494 |
| macro F1 (tuned offsets, in-sample) | 0.9577 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9460 – 0.9679 |
| offsets (log-prob) | [0.0, -0.05, -2.65] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9635 |   0.9724 | 0.9679 |      1736 |
| JAILBREAK        |      0.9509 |   0.9366 | 0.9437 |       868 |
| PROMPT_INJECTION |      0.9689 |   0.9541 | 0.9614 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.9473 |      0.9829 |         0.9943 |                0.8649 |
| fr         |  700 |     0.9675 |      0.9862 |         0.9885 |                0.9278 |
| it         |  559 |     0.9833 |      0.9899 |         0.9857 |                0.9744 |
| pt         | 1260 |     0.9423 |      0.9448 |         0.8877 |                0.9944 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1688 |               42 |                       6 |
| true_JAILBREAK        |            55 |              813 |                       0 |
| true_PROMPT_INJECTION |             9 |                0 |                     187 |

## Test prediction distribution
- BENIGN: 62.7%
- JAILBREAK: 30.6%
- PROMPT_INJECTION: 6.7%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9494 |   11.0241 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.344 |      0.0187 |    2100 |         1 |         7.6197 |
