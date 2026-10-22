# mmbert-base-cw0-full

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [0] / 5 · 13.64 min · updated 2026-10-22T09:40:48+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9983 |
| macro F1 (tuned offsets, in-sample) | 0.9991 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9978 – 1.0000 |
| offsets (log-prob) | [0.0, 2.35, 0.0] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9994 |   0.9988 | 0.9991 |      1736 |
| JAILBREAK        |      0.9977 |   0.9988 | 0.9983 |       868 |
| PROMPT_INJECTION |      1      |   1      | 1      |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     1      |      1      |         1      |                     1 |
| fr         |  700 |     1      |      1      |         1      |                     1 |
| it         |  559 |     1      |      1      |         1      |                     1 |
| pt         | 1260 |     0.9981 |      0.9981 |         0.9962 |                     1 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1734 |                2 |                       0 |
| true_JAILBREAK        |             1 |              867 |                       0 |
| true_PROMPT_INJECTION |             0 |                0 |                     196 |

## Test prediction distribution
- BENIGN: 62.5%
- JAILBREAK: 30.6%
- PROMPT_INJECTION: 6.9%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     14000 |    2800 |     0.9983 |   13.3228 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.832 |      0.0194 |    2625 |         1 |         7.6197 |
