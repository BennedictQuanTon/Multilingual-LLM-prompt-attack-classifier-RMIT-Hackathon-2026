# Team GenData — RMIT Hackathon 2026

[![CI](https://github.com/BennedictQuanTon/Team-GenData-RMIT-Hackathon-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/BennedictQuanTon/Team-GenData-RMIT-Hackathon-2026/actions/workflows/ci.yml)

**Challenge 1 — Multilingual prompt-attack classification** ([Kaggle](https://www.kaggle.com/competitions/rmit-hackathon-2026)):
phân loại prompt tiếng pt / fr / it / en thành `BENIGN`, `JAILBREAK`, `PROMPT_INJECTION`. Thước đo: **macro F1**.
Deadline **22/10/2026 18:00 giờ VN** (11:00 UTC) · 6 lượt nộp/ngày · chỉ được dùng **1 backbone** trong danh sách BTC duyệt.

| tài liệu | nội dung |
|---|---|
| [docs/PROBLEM.md](docs/PROBLEM.md) | đề bài, dữ liệu, luật, leaderboard hiện tại |
| [docs/PLAN.md](docs/PLAN.md) | **kế hoạch thi**: hướng tiếp cận, timeline, phân vai, rủi ro |
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | cách chạy local / Kaggle / ensemble / nộp bài / CI |
| [experiments/LOG.md](experiments/LOG.md) | nhật ký thí nghiệm của team |

## Bắt đầu nhanh

```bash
make setup                                   # uv env (Python 3.11) + pre-commit
make data && make eda && make baseline       # cần Join competition + ~/.kaggle/kaggle.json
make push CONFIG=configs/xlmr-large.yaml FOLDS=0   # chạy fold 0 trên Kaggle T4×2
make help                                    # mọi lệnh
```

## Pipeline

1. **Split** — 5-fold `StratifiedGroupKFold` theo *label × language*, prompt trùng nhau luôn chung fold; fingerprint của split
   đảm bảo mọi run của mọi thành viên ghép được với nhau.
2. **Model** — fine-tune 1 backbone được duyệt (`AutoModelForSequenceClassification`, 3 lớp), cắt đầu + đuôi cho prompt dài,
   cross-entropy có trọng số lớp; đường LLM dùng 4-bit + LoRA + đầu phân loại.
3. **Quyết định** — offset log-prob theo lớp tối ưu macro F1 trên OOF (đánh giá trung thực bằng cross-fit).
4. **Ensemble** — trung bình fold / seed / config của cùng backbone; artifact lưu sau mỗi fold.
5. **Nộp** — notebook `RMIT_SGS_GenData_C1` build từ git; `tools/submit.py` validate + log mọi lượt nộp.

## Cấu trúc

```
configs/                 1 file YAML = 1 thí nghiệm (model, lr, epochs, max_len, folds…)
src/rmitguard/
  constants.py           nhãn, danh sách model được duyệt + handle Kaggle
  config.py              ExperimentConfig (bắt lỗi gõ sai key, model không được duyệt)
  data.py                load, chuẩn hoá, fold + fingerprint
  models/tfidf.py        baseline CPU (chỉ để chẩn đoán)
  models/transformer.py  encoder full fine-tune / LLM QLoRA, Trainer có class weight
  postprocess.py         offset theo lớp cho macro F1
  runner.py              K-fold → oof.npy, test.npy, report.md, submission.csv
  ensemble.py            ghép fold subset / nhiều run (kiểm tra split + luật 1 backbone)
  eda.py, metrics.py, submission.py, cli.py
tools/
  build_kernel.py        repo → notebook Kaggle tự chứa + kernel-metadata.json
  submit.py              nộp bài có kiểm tra (validate, đếm lượt, log)
  smoke_transformer.py   chạy thật HF Trainer với XLM-R tí hon trên CPU (CI)
kaggle/team.yaml         university / campus / team cho tên notebook
tests/                   pytest trên dữ liệu tổng hợp (không cần data thật)
.github/                 CI, push/submit Kaggle, template issue/PR
```

## Quy ước

- Không commit dữ liệu cuộc thi (CC BY-NC-SA), trọng số, `kaggle.json` — pre-commit và CI chặn.
- Repo để **Private** đến hết cuộc thi (luật Kaggle cấm chia sẻ code ngoài team).
- So sánh thí nghiệm bằng **CV cross-fitted**, không bằng public LB (chỉ 30% test).
