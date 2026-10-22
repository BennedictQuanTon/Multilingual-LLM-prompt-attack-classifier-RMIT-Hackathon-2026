# granite-2b-lora-s2

- model: `granite-2b` (ibm-granite/granite-3.3-2b-instruct) · split `10a79e50dde8`
- folds done: [0] / 5 · 95.27 min · updated 2026-10-22T09:56:36+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9479 |
| macro F1 (tuned offsets, in-sample) | 0.9496 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9350 – 0.9634 |
| offsets (log-prob) | [0.0, -0.5, 0.0] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9707 |   0.9735 | 0.9721 |      1736 |
| JAILBREAK        |      0.962  |   0.962  | 0.962  |       868 |
| PROMPT_INJECTION |      0.9267 |   0.9031 | 0.9147 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.9247 |      0.9799 |         0.9942 |                0.8    |
| fr         |  700 |     0.9526 |      0.9806 |         0.9838 |                0.8936 |
| it         |  559 |     0.9303 |      0.974  |         0.9857 |                0.8312 |
| pt         | 1260 |     0.9619 |      0.9648 |         0.9323 |                0.9886 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1690 |               33 |                      13 |
| true_JAILBREAK        |            32 |              835 |                       1 |
| true_PROMPT_INJECTION |            19 |                0 |                     177 |

## Test prediction distribution
- BENIGN: 62.3%
- JAILBREAK: 30.9%
- PROMPT_INJECTION: 6.9%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                                 |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9479 |   94.8014 | /kaggle/input/models/aisuko/granite-3.3-2b-instruct/transformers/default/1 |       143.799 |      0.0413 |     700 |         1 |         3.3556 |
