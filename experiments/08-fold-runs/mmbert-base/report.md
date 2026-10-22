# mmbert-base

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [3, 4] / 5 · 24.41 min · updated 2026-10-22T03:51:58+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 40.0% |
| macro F1 (argmax) | 0.9646 |
| macro F1 (tuned offsets, in-sample) | 0.9668 |
| **macro F1 (tuned, cross-fitted) ← trust this** | **0.9642** |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9565 – 0.9762 |
| offsets (log-prob) | [0.0, -1.1, 1.3] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9737 |   0.9804 | 0.977  |      3472 |
| JAILBREAK        |      0.9679 |   0.9556 | 0.9617 |      1736 |
| PROMPT_INJECTION |      0.9641 |   0.9592 | 0.9616 |       392 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  559 |     0.9548 |      0.987  |         0.9943 |                0.8831 |
| fr         | 1400 |     0.9766 |      0.9914 |         0.9954 |                0.943  |
| it         | 1121 |     0.9711 |      0.9856 |         0.9842 |                0.9434 |
| pt         | 2520 |     0.9619 |      0.9632 |         0.9253 |                0.9972 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          3404 |               54 |                      14 |
| true_JAILBREAK        |            77 |             1659 |                       0 |
| true_PROMPT_INJECTION |            15 |                1 |                     376 |

## Test prediction distribution
- BENIGN: 62.5%
- JAILBREAK: 30.3%
- PROMPT_INJECTION: 7.2%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      3 |     11200 |    2800 |     0.9671 |   11.886  | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       108.058 |      0.0197 |    2100 |         1 |         7.6197 |
|      4 |     11200 |    2800 |     0.9621 |   12.0952 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       108.157 |      0.0196 |    2100 |         1 |         7.6197 |
