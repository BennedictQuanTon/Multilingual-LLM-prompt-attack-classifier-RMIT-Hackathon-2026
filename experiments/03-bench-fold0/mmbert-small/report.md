# mmbert-small

- model: `mmbert-small` (jhu-clsp/mmBERT-small) · split `10a79e50dde8`
- folds done: [0] / 5 · 8.22 min · updated 2026-10-21T22:35:06+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9544 |
| macro F1 (tuned offsets, in-sample) | 0.9593 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9482 – 0.9681 |
| offsets (log-prob) | [0.0, 0.5, -3.0] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9598 |   0.977  | 0.9683 |      1736 |
| JAILBREAK        |      0.9528 |   0.9297 | 0.9411 |       868 |
| PROMPT_INJECTION |      0.9946 |   0.9439 | 0.9686 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.9419 |      0.9801 |         0.9886 |                0.8571 |
| fr         |  700 |     0.9698 |      0.9851 |         0.9767 |                0.9474 |
| it         |  559 |     0.9909 |      0.9942 |         0.9914 |                0.987  |
| pt         | 1260 |     0.9422 |      0.945  |         0.8871 |                0.9943 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1696 |               39 |                       1 |
| true_JAILBREAK        |            61 |              807 |                       0 |
| true_PROMPT_INJECTION |            10 |                1 |                     185 |

## Test prediction distribution
- BENIGN: 63.3%
- JAILBREAK: 30.2%
- PROMPT_INJECTION: 6.4%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                      |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:----------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9544 |    7.8466 | /kaggle/input/models/aisuko/mmbert-small/transformers/default/1 |       107.344 |      0.0187 |    1050 |         1 |         8.9204 |
