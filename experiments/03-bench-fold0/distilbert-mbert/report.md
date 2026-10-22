# distilbert-mbert

- model: `distilbert-mbert` (distilbert/distilbert-base-multilingual-cased) · split `10a79e50dde8`
- folds done: [0] / 5 · 5.59 min · updated 2026-10-21T22:32:28+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9201 |
| macro F1 (tuned offsets, in-sample) | 0.9237 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9079 – 0.9368 |
| offsets (log-prob) | [0.0, 0.6, 0.5] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9545 |   0.9309 | 0.9425 |      1736 |
| JAILBREAK        |      0.8799 |   0.9286 | 0.9036 |       868 |
| PROMPT_INJECTION |      0.9372 |   0.9133 | 0.9251 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.8912 |      0.9687 |         0.9829 |                0.7222 |
| fr         |  700 |     0.9559 |      0.9828 |         0.9817 |                0.9032 |
| it         |  559 |     0.9577 |      0.9798 |         0.9796 |                0.9136 |
| pt         | 1260 |     0.8979 |      0.8962 |         0.8145 |                0.9831 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1616 |              108 |                      12 |
| true_JAILBREAK        |            62 |              806 |                       0 |
| true_PROMPT_INJECTION |            15 |                2 |                     179 |

## Test prediction distribution
- BENIGN: 60.7%
- JAILBREAK: 32.6%
- PROMPT_INJECTION: 6.7%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                                            |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:--------------------------------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9201 |    5.2356 | /kaggle/input/models/aisuko/distilbert-base-multilingual-cased/transformers/default/1 |       122.305 |       0.026 |    1050 |         1 |         5.2276 |
