# mmbert-base-lang

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [0] / 5 · 11.3 min · updated 2026-10-22T00:33:23+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9592 |
| macro F1 (tuned offsets, in-sample) | 0.9650 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9549 – 0.9735 |
| offsets (log-prob) | [0.0, 2.9, 2.35] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9796 |   0.9672 | 0.9733 |      1736 |
| JAILBREAK        |      0.9418 |   0.9689 | 0.9551 |       868 |
| PROMPT_INJECTION |      0.9741 |   0.9592 | 0.9666 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.9644 |      0.9857 |         0.9886 |                0.9189 |
| fr         |  700 |     0.9778 |      0.9908 |         0.9931 |                0.9495 |
| it         |  559 |     0.9734 |      0.987  |         0.9857 |                0.9474 |
| pt         | 1260 |     0.9543 |      0.9545 |         0.9141 |                0.9944 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1679 |               52 |                       5 |
| true_JAILBREAK        |            27 |              841 |                       0 |
| true_PROMPT_INJECTION |             8 |                0 |                     188 |

## Test prediction distribution
- BENIGN: 61.8%
- JAILBREAK: 31.3%
- PROMPT_INJECTION: 6.9%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9592 |   11.1125 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       110.288 |      0.0187 |    2100 |         1 |         7.6197 |
