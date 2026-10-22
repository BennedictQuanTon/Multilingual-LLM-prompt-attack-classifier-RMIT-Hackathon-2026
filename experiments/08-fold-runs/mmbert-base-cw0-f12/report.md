# mmbert-base-cw0

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [1, 2] / 5 · 24.42 min · updated 2026-10-22T04:41:36+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 40.0% |
| macro F1 (argmax) | 0.9683 |
| macro F1 (tuned offsets, in-sample) | 0.9710 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9705** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9608 – 0.9804 |
| offsets (log-prob) | [0.0, 1.45, 0.7] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9829 |   0.9775 | 0.9802 |      3472 |
| JAILBREAK        |      0.957  |   0.9735 | 0.9652 |      1736 |
| PROMPT_INJECTION |      0.9816 |   0.9541 | 0.9677 |       392 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  560 |     0.9786 |      0.9957 |         0.9914 |                0.9487 |
| fr         | 1400 |     0.9845 |      0.9942 |         0.9908 |                0.9684 |
| it         | 1120 |     0.9734 |      0.9899 |         0.9942 |                0.9359 |
| pt         | 2520 |     0.9609 |      0.9646 |         0.9326 |                0.9857 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          3394 |               71 |                       7 |
| true_JAILBREAK        |            46 |             1690 |                       0 |
| true_PROMPT_INJECTION |            13 |                5 |                     374 |

## Test prediction distribution
- BENIGN: 61.3%
- JAILBREAK: 31.8%
- PROMPT_INJECTION: 6.9%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      1 |     11200 |    2800 |     0.9655 |   12.0028 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.811 |      0.0196 |    2100 |         1 |         7.6197 |
|      2 |     11200 |    2800 |     0.9711 |   12.1503 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.789 |      0.0196 |    2100 |         1 |         7.6197 |
