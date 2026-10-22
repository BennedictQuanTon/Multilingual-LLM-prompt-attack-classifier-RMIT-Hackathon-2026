# granite-2b-lora

- model: `granite-2b` (ibm-granite/granite-3.3-2b-instruct) · split `10a79e50dde8`
- folds done: [3] / 5 · 84.68 min · updated 2026-10-22T08:20:00+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9437 |
| macro F1 (tuned offsets, in-sample) | 0.9476 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9334 – 0.9606 |
| offsets (log-prob) | [0.0, -0.15, 0.8] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9716 |   0.9666 | 0.9691 |      1735 |
| JAILBREAK        |      0.9424 |   0.9608 | 0.9515 |       868 |
| PROMPT_INJECTION |      0.9418 |   0.9036 | 0.9223 |       197 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  279 |     0.9375 |      0.9797 |         0.9827 |                0.85   |
| fr         |  700 |     0.9305 |      0.9748 |         0.9746 |                0.8421 |
| it         |  561 |     0.964  |      0.9813 |         0.9773 |                0.9333 |
| pt         | 1260 |     0.952  |      0.958  |         0.9208 |                0.9773 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1677 |               49 |                       9 |
| true_JAILBREAK        |            32 |              834 |                       2 |
| true_PROMPT_INJECTION |            17 |                2 |                     178 |

## Test prediction distribution
- BENIGN: 62.3%
- JAILBREAK: 31.0%
- PROMPT_INJECTION: 6.7%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                                 |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      3 |     11200 |    2800 |     0.9437 |   84.2998 | /kaggle/input/models/aisuko/granite-3.3-2b-instruct/transformers/default/1 |       144.432 |      0.0426 |     700 |         1 |         3.3556 |
