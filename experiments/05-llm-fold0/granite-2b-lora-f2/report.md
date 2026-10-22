# granite-2b-lora

- model: `granite-2b` (ibm-granite/granite-3.3-2b-instruct) · split `10a79e50dde8`
- folds done: [2] / 5 · 81.2 min · updated 2026-10-22T08:53:17+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.6278 |
| macro F1 (tuned offsets, in-sample) | 0.7083 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.6780 – 0.7423 |
| offsets (log-prob) | [0.0, 3.0, -1.15] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.7948 |   0.9401 | 0.8614 |      1735 |
| JAILBREAK        |      0.8522 |   0.5972 | 0.7023 |       869 |
| PROMPT_INJECTION |      0.6763 |   0.4796 | 0.5612 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  280 |     0.6151 |      0.8619 |         0.7333 |                0.25   |
| fr         |  700 |     0.7456 |      0.8908 |         0.7835 |                0.5625 |
| it         |  560 |     0.7124 |      0.8992 |         0.8535 |                0.3846 |
| pt         | 1260 |     0.7107 |      0.8296 |         0.5687 |                0.7338 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1631 |               84 |                      20 |
| true_JAILBREAK        |           325 |              519 |                      25 |
| true_PROMPT_INJECTION |            96 |                6 |                      94 |

## Test prediction distribution
- BENIGN: 79.4%
- JAILBREAK: 13.6%
- PROMPT_INJECTION: 6.9%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                                 |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      2 |     11200 |    2800 |     0.6278 |   80.8167 | /kaggle/input/models/aisuko/granite-3.3-2b-instruct/transformers/default/1 |       144.255 |      0.0409 |     700 |         1 |         3.3556 |
