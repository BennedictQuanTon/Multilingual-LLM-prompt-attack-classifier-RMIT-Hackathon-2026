# RMIT Hackathon 2026 — Team GenData

[![CI](https://github.com/BennedictQuanTon/Team-GenData-RMIT-Hackathon-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/BennedictQuanTon/Team-GenData-RMIT-Hackathon-2026/actions/workflows/ci.yml)

**RMIT GenAI & Cybersecurity Hackathon 2026** — *Security × Generative AI × Low-Resource Languages*.
Tổ chức 22–23/10/2026 tại Australia, Việt Nam và Malaysia. Đơn vị tham gia cùng RMIT: Taylor's University, Tide, Design Village, Katalon.

## Cuộc thi

Hackathon gồm nhiều challenge. **Challenge 1** chạy trên Kaggle với leaderboard; các challenge còn lại chạy trên **RMIT RACE Cloud**.

| | |
|---|---|
| Trang thi | https://www.kaggle.com/competitions/rmit-hackathon-2026 |
| Website | https://rmit-hackathon.com/ |
| Host | Bowen (Kaggle: `aisuko`) · notebook mẫu: [RMIT_SGS_demo_C1](https://www.kaggle.com/code/aisuko/rmit-sgs-demo-c1) |
| Hạn nộp Challenge 1 | **22/10/2026 18:00 giờ Việt Nam** (11:00 UTC) |
| Giải | Kudos (không có điểm / huy chương Kaggle) |

## Challenge 1 — Multilingual Protector Model

Đóng vai **AI Protector**: đọc một prompt gửi tới LLM và quyết định nó là

| nhãn | ý nghĩa | tỉ lệ train |
|---|---|---|
| `BENIGN` | yêu cầu bình thường, an toàn | 62% |
| `JAILBREAK` | cố thuyết phục model bỏ luật an toàn (nhập vai, "DAN", "developer mode"…) | 31% |
| `PROMPT_INJECTION` | cố chiếm quyền chỉ dẫn của model ("ignore previous instructions", lộ system prompt…) | 7% |

**Vì sao khó:** prompt viết bằng **tiếng Bồ Đào Nha, Pháp, Ý** (và một phần tiếng Anh). Các ngôn ngữ này giàu tài nguyên cho NLP nói chung
nhưng **nghèo tài nguyên cho AI safety**: hầu hết dataset prompt injection / guardrail được xây cho tiếng Anh,
nên bộ lọc an toàn hay bỏ lọt tấn công viết bằng ngôn ngữ khác. Mục tiêu của đề là lấp khoảng trống đó.

### Dữ liệu

| file | số dòng | cột |
|---|---|---|
| `train.csv` | 14.000 | `id`, `text`, `language`, `label` |
| `test.csv` | 6.000 | `id`, `text`, `language` |
| `sample_submission.csv` | 6.000 | `id`, `label` |

`language` ∈ {`pt`, `fr`, `it`, `en`}; `text` dài 10–18.550 ký tự. Giấy phép dữ liệu CC BY-NC-SA 4.0.

### Đánh giá

**Macro F1**: F1 tính riêng cho từng lớp rồi lấy trung bình không trọng số, nên lớp hiếm `PROMPT_INJECTION` nặng ngang `BENIGN`.
Đoán toàn `BENIGN` được ~0.26, hoàn hảo là 1.0. Public leaderboard chấm 30% test, private leaderboard (quyết định thứ hạng) chấm 70% còn lại.

File nộp: đúng 6.000 dòng + header.

```
id,label
t000006,BENIGN
t000007,PROMPT_INJECTION
t000008,JAILBREAK
```

### Luật chính

- Fine-tune **một** model trong danh sách được duyệt (XLM-RoBERTa base/large, mmBERT small/base, DistilBERT multilingual, RemBERT,
  DeBERTa-v3 prompt-injection-v2, Wolf Defender, Granite 3.3 2B Instruct, Mistral 7B Instruct v0.3); không dùng backbone pretrained nào khác.
- Ghi tên model đã chọn ở đầu notebook; notebook đặt tên `<university>_<campus>_<team>_C1` → của team: **`RMIT_SGS_GenData_C1`**.
- 6 lượt nộp mỗi ngày, team tối đa 4 người. Có điểm Challenge 1 ngay khi có submission hợp lệ đầu tiên.

## Tài liệu của team

| | |
|---|---|
| [Plan.md](Plan.md) | **tình hình, luật, chiến lược, việc từng người tối nay và ngày thi** |
| [docs/PROBLEM.md](docs/PROBLEM.md) | đề bài chi tiết |
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | cách chạy local / Kaggle / ensemble / nộp bài / CI |
| [experiments/LOG.md](experiments/LOG.md) | nhật ký thí nghiệm |

Bắt đầu: `make setup` → `make data` → `make baseline` → `make help`.
