# deberta-pi-v2

- model: `deberta-pi-v2` (protectai/deberta-v3-base-prompt-injection-v2) · split `10a79e50dde8`
- folds done: [0] / 5 · 15.9 min · updated 2026-10-21T16:15:25+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9484 |
| macro F1 (tuned offsets, in-sample) | 0.9518 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9396 – 0.9635 |
| offsets (log-prob) | [0.0, 1.6, -1.6] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9665 |   0.9649 | 0.9657 |      1736 |
| JAILBREAK        |      0.933  |   0.9459 | 0.9394 |       868 |
| PROMPT_INJECTION |      0.9733 |   0.9286 | 0.9504 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.9477 |      0.9886 |         0.9827 |                0.8718 |
| fr         |  700 |     0.9751 |      0.9885 |         0.9885 |                0.9485 |
| it         |  559 |     0.9589 |      0.9827 |         0.9774 |                0.9167 |
| pt         | 1260 |     0.9363 |      0.9403 |         0.8858 |                0.9829 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1675 |               58 |                       3 |
| true_JAILBREAK        |            45 |              821 |                       2 |
| true_PROMPT_INJECTION |            13 |                1 |                     182 |

## Test prediction distribution
- BENIGN: 62.0%
- JAILBREAK: 31.8%
- PROMPT_INJECTION: 6.3%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                                             |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:---------------------------------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9484 |   15.7173 | /kaggle/input/models/aisuko/deberta-v3-base-prompt-injection-v2/transformers/default/1 |       139.218 |      0.0369 |    2100 |         1 |        13.3998 |
