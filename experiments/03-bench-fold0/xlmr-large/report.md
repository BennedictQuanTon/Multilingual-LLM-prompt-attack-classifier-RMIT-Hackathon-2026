# xlmr-large

- model: `xlmr-large` (FacebookAI/xlm-roberta-large) · split `10a79e50dde8`
- folds done: [0] / 5 · 35.03 min · updated 2026-10-21T23:07:53+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9479 |
| macro F1 (tuned offsets, in-sample) | 0.9530 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9417 – 0.9648 |
| offsets (log-prob) | [0.0, -0.35, -1.7] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9716 |   0.9649 | 0.9682 |      1736 |
| JAILBREAK        |      0.9351 |   0.9631 | 0.9489 |       868 |
| PROMPT_INJECTION |      0.978  |   0.9082 | 0.9418 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.941  |      0.9829 |         0.9831 |                0.8571 |
| fr         |  700 |     0.9636 |      0.9828 |         0.9816 |                0.9263 |
| it         |  559 |     0.9726 |      0.9827 |         0.9742 |                0.961  |
| pt         | 1260 |     0.9407 |      0.9502 |         0.9127 |                0.9591 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1675 |               57 |                       4 |
| true_JAILBREAK        |            32 |              836 |                       0 |
| true_PROMPT_INJECTION |            17 |                1 |                     178 |

## Test prediction distribution
- BENIGN: 61.5%
- JAILBREAK: 32.3%
- PROMPT_INJECTION: 6.2%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                           |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9479 |    34.842 | /kaggle/input/models/aisuko/xlm-roberta-large/transformers/default/1 |        113.22 |      0.0196 |    1400 |         1 |           7.12 |
