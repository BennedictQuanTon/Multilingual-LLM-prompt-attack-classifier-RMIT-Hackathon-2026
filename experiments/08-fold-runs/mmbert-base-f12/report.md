# mmbert-base

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [1, 2] / 5 · 24.67 min · updated 2026-10-22T03:52:13+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 40.0% |
| macro F1 (argmax) | 0.9696 |
| macro F1 (tuned offsets, in-sample) | 0.9702 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9690** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9607 – 0.9800 |
| offsets (log-prob) | [0.0, -0.7, 0.0] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9784 |   0.9798 | 0.9791 |      3472 |
| JAILBREAK        |      0.9616 |   0.966  | 0.9638 |      1736 |
| PROMPT_INJECTION |      0.9842 |   0.9515 | 0.9676 |       392 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  560 |     0.9805 |      0.9957 |         0.9971 |                0.9487 |
| fr         | 1400 |     0.984  |      0.9942 |         0.9943 |                0.9634 |
| it         | 1120 |     0.9768 |      0.9921 |         0.9971 |                0.9412 |
| pt         | 2520 |     0.9572 |      0.9612 |         0.9248 |                0.9857 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          3402 |               64 |                       6 |
| true_JAILBREAK        |            59 |             1677 |                       0 |
| true_PROMPT_INJECTION |            16 |                3 |                     373 |

## Test prediction distribution
- BENIGN: 63.2%
- JAILBREAK: 30.1%
- PROMPT_INJECTION: 6.7%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      1 |     11200 |    2800 |     0.9649 |   12.1408 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.811 |      0.0196 |    2100 |         1 |         7.6197 |
|      2 |     11200 |    2800 |     0.9744 |   12.1099 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.789 |      0.0196 |    2100 |         1 |         7.6197 |
