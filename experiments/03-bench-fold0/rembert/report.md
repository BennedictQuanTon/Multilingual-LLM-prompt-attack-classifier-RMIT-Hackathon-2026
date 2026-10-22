# rembert

- model: `rembert` (google/rembert) · split `10a79e50dde8`
- folds done: [0] / 5 · 46.77 min · updated 2026-10-21T23:21:58+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9443 |
| macro F1 (tuned offsets, in-sample) | 0.9549 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9418 – 0.9663 |
| offsets (log-prob) | [0.0, 1.75, -2.8] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9655 |   0.9666 | 0.966  |      1736 |
| JAILBREAK        |      0.9371 |   0.9435 | 0.9403 |       868 |
| PROMPT_INJECTION |      0.9787 |   0.9388 | 0.9583 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.9172 |      0.9742 |         0.9775 |                0.8    |
| fr         |  700 |     0.9828 |      0.9908 |         0.9885 |                0.9691 |
| it         |  559 |     0.9673 |      0.9841 |         0.9828 |                0.9351 |
| pt         | 1260 |     0.941  |      0.9424 |         0.8863 |                0.9943 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1678 |               54 |                       4 |
| true_JAILBREAK        |            49 |              819 |                       0 |
| true_PROMPT_INJECTION |            11 |                1 |                     184 |

## Test prediction distribution
- BENIGN: 62.1%
- JAILBREAK: 31.5%
- PROMPT_INJECTION: 6.4%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                 |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:-----------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9443 |   46.5953 | /kaggle/input/models/aisuko/rembert/transformers/default/1 |       108.476 |      0.0185 |    1050 |         1 |        10.6552 |
