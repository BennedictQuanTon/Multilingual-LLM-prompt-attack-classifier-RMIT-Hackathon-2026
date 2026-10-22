# granite-2b-lora

- model: `granite-2b` (ibm-granite/granite-3.3-2b-instruct) · split `10a79e50dde8`
- folds done: [0] / 5 · 85.08 min · updated 2026-10-22T05:03:54+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9525 |
| macro F1 (tuned offsets, in-sample) | 0.9563 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9423 – 0.9689 |
| offsets (log-prob) | [0.0, 0.4, -2.85] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9803 |   0.9764 | 0.9784 |      1736 |
| JAILBREAK        |      0.9611 |   0.9677 | 0.9644 |       868 |
| PROMPT_INJECTION |      0.9239 |   0.9286 | 0.9262 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.9004 |      0.9798 |         0.9714 |                0.75   |
| fr         |  700 |     0.967  |      0.9896 |         0.9931 |                0.9184 |
| it         |  559 |     0.9478 |      0.9855 |         0.9828 |                0.875  |
| pt         | 1260 |     0.9673 |      0.9686 |         0.9389 |                0.9943 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1695 |               31 |                      10 |
| true_JAILBREAK        |            23 |              840 |                       5 |
| true_PROMPT_INJECTION |            11 |                3 |                     182 |

## Test prediction distribution
- BENIGN: 62.3%
- JAILBREAK: 31.1%
- PROMPT_INJECTION: 6.6%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                                 |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9525 |   84.6502 | /kaggle/input/models/aisuko/granite-3.3-2b-instruct/transformers/default/1 |       143.799 |      0.0413 |     700 |         1 |         3.3556 |
