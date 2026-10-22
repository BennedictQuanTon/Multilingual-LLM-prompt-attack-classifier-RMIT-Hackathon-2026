# granite-2b-lora

- model: `granite-2b` (ibm-granite/granite-3.3-2b-instruct) · split `10a79e50dde8`
- folds done: [2] / 5 · 91.95 min · updated 2026-10-22T07:25:26+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 17.5% |
| macro F1 (argmax) | 0.3678 |
| macro F1 (tuned offsets, in-sample) | 0.4045 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.3764 – 0.4299 |
| offsets (log-prob) | [0.0, 3.0, 2.55] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.6483 |   0.7766 | 0.7067 |      1531 |
| JAILBREAK        |      0.4416 |   0.2584 | 0.326  |       747 |
| PROMPT_INJECTION |      0.1771 |   0.1845 | 0.1808 |       168 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  260 |     0.4537 |      0.7571 |         0.4706 |                0.1333 |
| fr         |  620 |     0.4243 |      0.72   |         0.2993 |                0.2535 |
| it         |  505 |     0.3588 |      0.6977 |         0.1941 |                0.1846 |
| pt         | 1061 |     0.4041 |      0.6905 |         0.3636 |                0.1582 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1189 |              216 |                     126 |
| true_JAILBREAK        |           536 |              193 |                      18 |
| true_PROMPT_INJECTION |           109 |               28 |                      31 |

## Test prediction distribution
- BENIGN: 77.6%
- JAILBREAK: 15.8%
- PROMPT_INJECTION: 6.6%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                                 |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      2 |     11200 |    2800 |     0.3558 |   91.5008 | /kaggle/input/models/aisuko/granite-3.3-2b-instruct/transformers/default/1 |       144.255 |      0.0409 |     700 |         1 |         3.3556 |
