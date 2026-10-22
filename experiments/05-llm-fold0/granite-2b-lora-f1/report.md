# granite-2b-lora

- model: `granite-2b` (ibm-granite/granite-3.3-2b-instruct) · split `10a79e50dde8`
- folds done: [1] / 5 · 91.17 min · updated 2026-10-22T07:24:39+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9416 |
| macro F1 (tuned offsets, in-sample) | 0.9473 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9299 – 0.9606 |
| offsets (log-prob) | [0.0, 1.7, -0.4] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9795 |   0.9632 | 0.9713 |      1737 |
| JAILBREAK        |      0.942  |   0.9735 | 0.9575 |       867 |
| PROMPT_INJECTION |      0.9133 |   0.9133 | 0.9133 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  280 |     0.9404 |      0.9827 |         0.9885 |                0.85   |
| fr         |  700 |     0.9489 |      0.9826 |         0.9774 |                0.8866 |
| it         |  560 |     0.9303 |      0.9725 |         0.9829 |                0.8354 |
| pt         | 1260 |     0.9559 |      0.9619 |         0.9285 |                0.9773 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1673 |               47 |                      17 |
| true_JAILBREAK        |            23 |              844 |                       0 |
| true_PROMPT_INJECTION |            12 |                5 |                     179 |

## Test prediction distribution
- BENIGN: 61.2%
- JAILBREAK: 32.0%
- PROMPT_INJECTION: 6.8%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                                 |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      1 |     11200 |    2800 |     0.9416 |   90.7146 | /kaggle/input/models/aisuko/granite-3.3-2b-instruct/transformers/default/1 |       144.177 |      0.0427 |     700 |         1 |         3.3556 |
