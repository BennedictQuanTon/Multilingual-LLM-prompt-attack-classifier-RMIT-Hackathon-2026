# mmbert-base-distill-full

- model: `mmbert-base` (jhu-clsp/mmBERT-base) · split `10a79e50dde8`
- folds done: [0] / 5 · 13.85 min · updated 2026-10-22T06:51:13+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9864 |
| macro F1 (tuned offsets, in-sample) | 0.9890 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9822 – 0.9943 |
| offsets (log-prob) | [0.0, -1.0, 0.0] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9891 |   0.9942 | 0.9917 |      1736 |
| JAILBREAK        |      0.9895 |   0.9816 | 0.9855 |       868 |
| PROMPT_INJECTION |      0.9948 |   0.9847 | 0.9897 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.9806 |      0.9943 |         1      |                0.9474 |
| fr         |  700 |     0.9939 |      0.9965 |         0.9954 |                0.9897 |
| it         |  559 |     0.9924 |      0.9957 |         0.9942 |                0.9873 |
| pt         | 1260 |     0.9865 |      0.9866 |         0.9729 |                1      |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1726 |                9 |                       1 |
| true_JAILBREAK        |            16 |              852 |                       0 |
| true_PROMPT_INJECTION |             3 |                0 |                     193 |

## Test prediction distribution
- BENIGN: 62.7%
- JAILBREAK: 30.5%
- PROMPT_INJECTION: 6.8%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                     |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     14000 |    2800 |     0.9864 |   13.6627 | /kaggle/input/models/aisuko/mmbert-base/transformers/default/1 |       107.832 |      0.0194 |    2625 |         1 |         7.6197 |
