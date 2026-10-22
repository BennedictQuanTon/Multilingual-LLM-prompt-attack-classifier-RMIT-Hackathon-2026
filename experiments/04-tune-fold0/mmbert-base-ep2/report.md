# mmbert-base-ep2

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [0] / 5 · 7.62 min · updated 2026-10-22T00:41:22+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9557 |
| macro F1 (tuned offsets, in-sample) | 0.9610 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9507 – 0.9709 |
| offsets (log-prob) | [0.0, 0.05, -2.55] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9663 |   0.9735 | 0.9699 |      1736 |
| JAILBREAK        |      0.9522 |   0.9412 | 0.9467 |       868 |
| PROMPT_INJECTION |      0.9741 |   0.9592 | 0.9666 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.9535 |      0.9829 |         0.9886 |                0.8889 |
| fr         |  700 |     0.968  |      0.9862 |         0.9885 |                0.9293 |
| it         |  559 |     0.9909 |      0.9942 |         0.9914 |                0.987  |
| pt         | 1260 |     0.945  |      0.9473 |         0.8932 |                0.9944 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1690 |               41 |                       5 |
| true_JAILBREAK        |            51 |              817 |                       0 |
| true_PROMPT_INJECTION |             8 |                0 |                     188 |

## Test prediction distribution
- BENIGN: 62.6%
- JAILBREAK: 30.7%
- PROMPT_INJECTION: 6.7%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9557 |     7.433 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.344 |      0.0187 |    1400 |         1 |         7.6197 |
