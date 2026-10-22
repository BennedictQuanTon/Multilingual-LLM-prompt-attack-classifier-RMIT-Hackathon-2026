# mmbert-base-s4

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [0, 1, 2] / 5 · 34.75 min · updated 2026-10-22T05:38:52+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 60.0% |
| macro F1 (argmax) | 0.9623 |
| macro F1 (tuned offsets, in-sample) | 0.9643 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9614** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9536 – 0.9736 |
| offsets (log-prob) | [0.0, 1.2, 2.35] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9815 |   0.9674 | 0.9744 |      5208 |
| JAILBREAK        |      0.9423 |   0.972  | 0.9569 |      2604 |
| PROMPT_INJECTION |      0.9673 |   0.9558 | 0.9615 |       588 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  841 |     0.9639 |      0.9885 |         0.9942 |                0.9091 |
| fr         | 2100 |     0.9793 |      0.9919 |         0.9908 |                0.955  |
| it         | 1679 |     0.9737 |      0.9889 |         0.9914 |                0.9407 |
| pt         | 3780 |     0.9523 |      0.9548 |         0.9156 |                0.9866 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          5038 |              151 |                      19 |
| true_JAILBREAK        |            73 |             2531 |                       0 |
| true_PROMPT_INJECTION |            22 |                4 |                     562 |

## Test prediction distribution
- BENIGN: 60.2%
- JAILBREAK: 32.5%
- PROMPT_INJECTION: 7.3%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9601 |   11.6221 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.344 |      0.0187 |    2100 |         1 |         7.6197 |
|      1 |     11200 |    2800 |     0.9623 |   11.3557 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.811 |      0.0196 |    2100 |         1 |         7.6197 |
|      2 |     11200 |    2800 |     0.9646 |   11.3586 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.789 |      0.0196 |    2100 |         1 |         7.6197 |
