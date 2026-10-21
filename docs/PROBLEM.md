# Đề bài — RMIT Hackathon 2026, Challenge 1

> Nguồn: trang cuộc thi https://www.kaggle.com/competitions/rmit-hackathon-2026 (lấy qua Kaggle API ngày 21/10/2026)
> và notebook mẫu của BTC https://www.kaggle.com/code/aisuko/rmit-sgs-demo-c1.

## Nhiệm vụ

**Multilingual Prompt-Attack Classification.** Cho một prompt tiếng **Bồ Đào Nha (pt), Pháp (fr), Ý (it) hoặc Anh (en)**,
phân vào 1 trong 3 lớp:

| label | ý nghĩa |
|---|---|
| `BENIGN` | yêu cầu bình thường, an toàn |
| `JAILBREAK` | cố thuyết phục model bỏ qua luật an toàn (nhập vai, "DAN", "developer mode"…) |
| `PROMPT_INJECTION` | cố chiếm quyền chỉ dẫn của model ("ignore previous instructions", lộ system prompt…) |

Ý đồ của đề: pt/fr/it là ngôn ngữ giàu tài nguyên cho NLP nói chung nhưng **nghèo tài nguyên cho AI safety**,
nên bộ lọc học từ tiếng Anh hay bỏ lọt. Phần lớn dữ liệu **không phải tiếng Anh**.

## Dữ liệu

| file | số dòng | cột |
|---|---|---|
| `train.csv` | 14,000 | `id`, `text`, `language`, `label` |
| `test.csv` | 6,000 | `id`, `text`, `language` |
| `sample_submission.csv` | 6,000 | `id`, `label` |

- `text`: 10 → 18,550 ký tự, **median 294** (đuôi dài: jailbreak nhập vai thường rất dài).
- **Lệch lớp:** `BENIGN` 62% · `JAILBREAK` 31% · `PROMPT_INJECTION` 7%.
- License dữ liệu: CC BY-NC-SA 4.0 → **không commit dữ liệu lên GitHub.**

## Thước đo: Macro F1

F1 tính riêng từng lớp rồi lấy trung bình **không trọng số** → lớp `PROMPT_INJECTION` (7%) nặng ngang `BENIGN` (62%).
Đoán toàn `BENIGN` được ~0.26. Hệ quả: **argmax không tối ưu macro F1** → phải chỉnh ngưỡng/offset theo lớp (đã có trong `postprocess.py`).

Public leaderboard = **30% test (~1,800 dòng, chỉ ~126 dòng PI)** → rất nhiễu (±0.01). Private = 70% còn lại quyết định thứ hạng.

## Luật quan trọng (đọc kỹ)

1. **Chỉ được dùng MỘT backbone trong danh sách được duyệt** ("no other pretrained backbone is allowed"),
   và phải **ghi model đã chọn ở đầu notebook** (`tools/build_kernel.py` tự làm).

   | key trong repo | model | loại | ghi chú BTC |
   |---|---|---|---|
   | `distilbert-mbert` | distilbert/distilbert-base-multilingual-cased | encoder | baseline nhanh nhất |
   | `mmbert-small` | jhu-clsp/mmBERT-small | encoder | nhanh, đa ngữ |
   | `mmbert-base` | jhu-clsp/mmBERT-base | encoder | đa ngữ mạnh, context dài |
   | `xlmr-base` | FacebookAI/xlm-roberta-base | encoder | đa ngữ mạnh |
   | `xlmr-large` | FacebookAI/xlm-roberta-large | encoder | encoder tốt nhất, tốn VRAM |
   | `rembert` | google/rembert | encoder | batch size 8 |
   | `deberta-pi-v2` | protectai/deberta-v3-base-prompt-injection-v2 | encoder (safety) | đã biết prompt injection (EN) |
   | `wolf-defender` | patronus-studio/wolf-defender-prompt-injection | encoder (safety) | đã biết prompt injection |
   | `granite-2b` | ibm-granite/granite-3.3-2b-instruct | LLM | LoRA / QLoRA |
   | `mistral-7b` | mistralai/Mistral-7B-Instruct-v0.3 | LLM | LoRA / QLoRA |

   ⚠️ **Chưa rõ** có được ensemble *nhiều* backbone khác nhau không. Repo mặc định **chặn** (cần `--allow-multi-backbone`).
   Ensemble nhiều fold / nhiều seed / nhiều config **của cùng một backbone** thì an toàn. → Hỏi BTC trên Discussion.
2. **6 lượt nộp / ngày** (reset 00:00 UTC = **07:00 giờ VN**). Team tối đa **4 người**.
3. **Deadline: 22/10/2026 11:00 UTC = 18:00 giờ Việt Nam** (22:00 Melbourne).
4. Tên notebook: `<university>_<campus>_<team>_C1` → của mình: **`RMIT_SGS_GenData_C1`**. Rules page ghi "Submit notebooks to get higher score"
   → nộp từ output của notebook (có sẵn `make submit KERNEL=...`).
5. Có điểm Challenge 1 ngay khi có **submission hợp lệ đầu tiên**; sau đó leo leaderboard.
6. Luật nền tảng Kaggle: **không chia sẻ code riêng ra ngoài team** trong thời gian thi. Repo GitHub phải để **Private** đến khi hết hạn.
7. Các challenge khác chạy trên **RMIT RACE Cloud** (không thuộc Kaggle).

## Bối cảnh leaderboard (21/10, ~20:00 giờ VN)

Top public: **0.97185** · hạng 5 ≈ 0.961 · notebook mẫu của BTC (`RMIT_HNI_sample_C1`) **0.90815** · all-BENIGN ≈ 0.26.
19 đội, 36 lượt nộp. → Mục tiêu: **≥ 0.975 CV**, và đừng đuổi theo public LB.
