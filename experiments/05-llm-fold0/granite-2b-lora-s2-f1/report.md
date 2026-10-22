# granite-2b-lora-s2

- model: `granite-2b` (ibm-granite/granite-3.3-2b-instruct) · split `10a79e50dde8`
- folds done: [1] / 5 · 95.92 min · updated 2026-10-22T09:57:16+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 19.9% |
| macro F1 (argmax) | 0.6827 |
| macro F1 (tuned offsets, in-sample) | 0.7396 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.7118 – 0.7644 |
| offsets (log-prob) | [0.0, -3.0, -2.85] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9732 |   0.6116 | 0.7512 |      1725 |
| JAILBREAK        |      0.5717 |   0.9792 | 0.7219 |       867 |
| PROMPT_INJECTION |      0.7064 |   0.7897 | 0.7458 |       195 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  273 |     0.7423 |      0.9198 |         0.8571 |                0.45   |
| fr         |  698 |     0.7707 |      0.8049 |         0.7673 |                0.74   |
| it         |  557 |     0.7299 |      0.7862 |         0.7834 |                0.62   |
| pt         | 1259 |     0.7334 |      0.6536 |         0.6564 |                0.8902 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1055 |              612 |                      58 |
| true_JAILBREAK        |            12 |              849 |                       6 |
| true_PROMPT_INJECTION |            17 |               24 |                     154 |

## Test prediction distribution
- BENIGN: 34.8%
- JAILBREAK: 56.0%
- PROMPT_INJECTION: 9.2%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                                 |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      1 |     11200 |    2800 |     0.6831 |   95.4592 | /kaggle/input/models/aisuko/granite-3.3-2b-instruct/transformers/default/1 |       144.177 |      0.0427 |     700 |         1 |         3.3556 |
