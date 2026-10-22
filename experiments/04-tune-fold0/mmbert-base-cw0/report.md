# mmbert-base-cw0

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [3, 4] / 5 · 24.46 min · updated 2026-10-22T04:41:59+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 40.0% |
| macro F1 (argmax) | 0.9676 |
| macro F1 (tuned offsets, in-sample) | 0.9705 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9690** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9627 – 0.9786 |
| offsets (log-prob) | [0.0, 1.4, 0.8] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9817 |   0.9755 | 0.9786 |      3472 |
| JAILBREAK        |      0.9548 |   0.9735 | 0.9641 |      1736 |
| PROMPT_INJECTION |      0.9842 |   0.9541 | 0.9689 |       392 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  559 |     0.9743 |      0.9943 |         0.9971 |                0.9315 |
| fr         | 1400 |     0.9782 |      0.992  |         0.9954 |                0.9474 |
| it         | 1121 |     0.977  |      0.9885 |         0.9871 |                0.9554 |
| pt         | 2520 |     0.9624 |      0.9632 |         0.9296 |                0.9943 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          3387 |               79 |                       6 |
| true_JAILBREAK        |            46 |             1690 |                       0 |
| true_PROMPT_INJECTION |            17 |                1 |                     374 |

## Test prediction distribution
- BENIGN: 60.5%
- JAILBREAK: 32.4%
- PROMPT_INJECTION: 7.1%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      3 |     11200 |    2800 |     0.9666 |   12.0395 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       108.058 |      0.0197 |    2100 |         1 |         7.6197 |
|      4 |     11200 |    2800 |     0.9686 |   12.1566 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       108.157 |      0.0196 |    2100 |         1 |         7.6197 |
