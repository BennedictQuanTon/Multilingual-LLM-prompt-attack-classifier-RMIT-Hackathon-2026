# granite-2b-lora

- model: `granite-2b` (ibm-granite/granite-3.3-2b-instruct) · split `10a79e50dde8`
- folds done: [4] / 5 · 90.82 min · updated 2026-10-22T08:26:08+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9343 |
| macro F1 (tuned offsets, in-sample) | 0.9402 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9238 – 0.9543 |
| offsets (log-prob) | [0.0, 0.3, -3.0] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9636 |   0.9603 | 0.9619 |      1737 |
| JAILBREAK        |      0.9279 |   0.9493 | 0.9385 |       868 |
| PROMPT_INJECTION |      0.9558 |   0.8872 | 0.9202 |       195 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  280 |     0.9516 |      0.9884 |         0.9775 |                0.8889 |
| fr         |  700 |     0.9391 |      0.977  |         0.9862 |                0.8542 |
| it         |  560 |     0.9504 |      0.9755 |         0.9718 |                0.9041 |
| pt         | 1260 |     0.9336 |      0.9417 |         0.8883 |                0.9708 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1668 |               61 |                       8 |
| true_JAILBREAK        |            44 |              824 |                       0 |
| true_PROMPT_INJECTION |            19 |                3 |                     173 |

## Test prediction distribution
- BENIGN: 61.6%
- JAILBREAK: 32.1%
- PROMPT_INJECTION: 6.4%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                                 |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      4 |     11200 |    2800 |     0.9343 |   90.4383 | /kaggle/input/models/aisuko/granite-3.3-2b-instruct/transformers/default/1 |       144.535 |      0.0432 |     700 |         1 |         3.3556 |
