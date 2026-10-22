# mmbert-base-ls

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [1, 2] / 5 · 24.56 min · updated 2026-10-22T05:06:25+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 40.0% |
| macro F1 (argmax) | 0.9632 |
| macro F1 (tuned offsets, in-sample) | 0.9639 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9608** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9523 – 0.9744 |
| offsets (log-prob) | [0.0, -0.5, -0.5] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9753 |   0.9781 | 0.9767 |      3472 |
| JAILBREAK        |      0.9619 |   0.9608 | 0.9614 |      1736 |
| PROMPT_INJECTION |      0.9635 |   0.9439 | 0.9536 |       392 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  560 |     0.9795 |      0.9986 |         0.9914 |                0.9487 |
| fr         | 1400 |     0.9727 |      0.9885 |         0.9908 |                0.9388 |
| it         | 1120 |     0.9686 |      0.9885 |         0.9942 |                0.9231 |
| pt         | 2520 |     0.9536 |      0.9601 |         0.924  |                0.9769 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          3396 |               63 |                      13 |
| true_JAILBREAK        |            67 |             1668 |                       1 |
| true_PROMPT_INJECTION |            19 |                3 |                     370 |

## Test prediction distribution
- BENIGN: 62.9%
- JAILBREAK: 30.3%
- PROMPT_INJECTION: 6.7%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      1 |     11200 |    2800 |     0.9569 |   12.0764 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.811 |      0.0196 |    2100 |         1 |         7.6197 |
|      2 |     11200 |    2800 |     0.9694 |   12.2171 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.789 |      0.0196 |    2100 |         1 |         7.6197 |
