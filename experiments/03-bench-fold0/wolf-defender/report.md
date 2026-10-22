# wolf-defender

- model: `wolf-defender` (patronus-studio/wolf-defender-prompt-injection) · split `10a79e50dde8`
- folds done: [0] / 5 · 11.48 min · updated 2026-10-21T15:59:21+00:00

## Out-of-fold (CV)
| metric | value |
|---|---|
| OOF coverage | 20.0% |
| macro F1 (argmax) | 0.9568 |
| macro F1 (tuned offsets, in-sample) | 0.9607 |
| public-LB-sized noise (95% CI, 1800 rows) | 0.9497 – 0.9718 |
| offsets (log-prob) | [0.0, 0.95, -0.95] |

### Per class
|                  |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| BENIGN           |      0.9735 |   0.9724 | 0.9729 |      1736 |
| JAILBREAK        |      0.954  |   0.9562 | 0.9551 |       868 |
| PROMPT_INJECTION |      0.9541 |   0.9541 | 0.9541 |       196 |

### Per language
| language   |    n |   macro_f1 |   f1_BENIGN |   f1_JAILBREAK |   f1_PROMPT_INJECTION |
|:-----------|-----:|-----------:|------------:|---------------:|----------------------:|
| en         |  281 |     0.935  |      0.9798 |         0.9831 |                0.8421 |
| fr         |  700 |     0.9668 |      0.9873 |         0.9931 |                0.92   |
| it         |  559 |     0.9784 |      0.9884 |         0.9857 |                0.961  |
| pt         | 1260 |     0.955  |      0.9565 |         0.914  |                0.9944 |

### Confusion (OOF, tuned)
|                       |   pred_BENIGN |   pred_JAILBREAK |   pred_PROMPT_INJECTION |
|:----------------------|--------------:|-----------------:|------------------------:|
| true_BENIGN           |          1688 |               39 |                       9 |
| true_JAILBREAK        |            38 |              830 |                       0 |
| true_PROMPT_INJECTION |             8 |                1 |                     187 |

## Test prediction distribution
- BENIGN: 61.9%
- JAILBREAK: 31.3%
- PROMPT_INJECTION: 6.9%

## Folds
|   fold |   n_train |   n_val |   macro_f1 |   minutes | model_path                                                                        |   mean_tokens |   truncated |   steps |   devices |   peak_vram_gb |
|-------:|----------:|--------:|-----------:|----------:|:----------------------------------------------------------------------------------|--------------:|------------:|--------:|----------:|---------------:|
|      0 |     11200 |    2800 |     0.9568 |   11.1395 | /kaggle/input/models/aisuko/wolf-defender-prompt-injection/transformers/default/1 |       107.344 |      0.0187 |    2100 |         1 |         7.6197 |
