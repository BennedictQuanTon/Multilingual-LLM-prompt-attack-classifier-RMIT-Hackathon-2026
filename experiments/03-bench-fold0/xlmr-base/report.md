# xlmr-base

- model: `xlmr-base` (FacebookAI/xlm-roberta-base) · split `10a79e50dde8`
- folds done: [0] / 5 · 10.66 min · updated 2026-10-21T15:58:32+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9297 |
| macro F1 (tuned offsets, in-sample) | 0.9329 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9174 – 0.9469 |
| offsets (log-prob) | [0.0, 1.1, -0.65] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9617 |   0.9395 | 0.9505 |      1736 |
| JAILBREAK        |      0.8942 |   0.9447 | 0.9188 |       868 |
| PROMPT_INJECTION |      0.9519 |   0.9082 | 0.9295 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.8991 |      0.9714 |         0.9831 |                0.7429 |
| fr         |  700 |     0.9605 |      0.9816 |         0.9815 |                0.9184 |
| it         |  559 |     0.963  |      0.9798 |         0.9742 |                0.9351 |
| pt         | 1260 |     0.9115 |      0.9145 |         0.8489 |                0.9711 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1631 |               96 |                       9 |
| true_JAILBREAK        |            48 |              820 |                       0 |
| true_PROMPT_INJECTION |            17 |                1 |                     178 |

## Test prediction distribution
- BENIGN: 60.5%
- JAILBREAK: 33.1%
- PROMPT_INJECTION: 6.5%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                          |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:--------------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9297 |   10.3213 | /kaggle/input/models/aisuko/xlm-roberta-base/transformers/default/1 |        113.22 |      0.0196 |    2100 |         1 |         6.9518 |
