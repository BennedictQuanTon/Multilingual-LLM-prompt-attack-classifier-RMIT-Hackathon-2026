# Kế hoạch thi — Team GenData (21/10 tối → 22/10 18:00)

**Mục tiêu:** có submission hợp lệ **ngay tối nay** (chắc điểm C1), sau đó đạt **macro F1 ≥ 0.975** trên CV,
nộp bản cuối trước **17:00 ngày 22/10** (deadline thật 18:00 giờ VN = 11:00 UTC).

## 1. Hướng tiếp cận (ngắn gọn)

Bài này là **phân loại văn bản 3 lớp, đa ngữ, lệch lớp, chấm bằng macro F1**. Thứ quyết định điểm, theo thứ tự:

1. **Chọn backbone đúng** (chỉ được 1): encoder đa ngữ mạnh nhất mà T4 chạy nổi → `xlmr-large` hoặc `mmbert-base`.
2. **Xử lý lớp hiếm `PROMPT_INJECTION`** (7% nhưng chiếm 1/3 điểm):
   class weight `sqrt` khi train + **chỉnh offset log-prob theo lớp trên OOF** (`postprocess.fit_bias`) — thường +1–3 điểm macro F1 miễn phí.
3. **Prompt dài:** cắt **đầu + đuôi** (384 + 128 token) để giữ cả phần dựng vai lẫn yêu cầu cuối.
4. **CV đáng tin:** 5-fold `StratifiedGroupKFold` theo *label × language*, gom các prompt trùng nhau vào cùng fold.
   Điểm tin được = **"macro F1 (tuned, cross-fitted)"** trong `report.md`, *không* phải public LB (1,800 dòng, nhiễu ±0.01).
5. **Ensemble trong luật:** trung bình 5 fold (+ thêm seed nếu kịp) của **cùng backbone**.

### Bài học từ 2025 (repo `Fine-tuning-Training`, notebook 04) áp vào năm nay

| 2025 | 2026 |
|---|---|
| Holdout + 5-fold CV, chọn mọi thứ trên OOF → hạng thật khớp ước lượng | Giữ nguyên: 5-fold OOF + cross-fit cho offset |
| Lưu `preds.npz` theo model, ghép lại không phải train lại | Lưu artifact **sau mỗi fold**; các thành viên chạy **fold khác nhau** rồi `ensemble` ghép lại |
| Ngân sách thời gian 11.3h trong notebook | `time_budget_hours` dừng trước fold không kịp |
| Qwen2.5-3B QLoRA + đầu phân loại là model đơn tốt nhất | Đường LLM = `granite-2b-lora` (stretch, chậm: ~2h/fold) |
| DeBERTa-base overfit sau ~2.5 epoch | Encoder: 3 epoch, theo dõi `eval_macro_f1` theo epoch trong `history.csv` |
| Lỗi còn lại: injection **không phải tiếng Anh**, prompt rất ngắn, role-play vô hại | Đúng trọng tâm năm nay → xem bảng *Per language* và *Confusion* sau mỗi run |
| transformers 5: loss log bị nhân với grad-accum | Trainer của mình tự chia đúng (`model_accepts_loss_kwargs=False`) |

### Kỳ vọng (ước lượng, cần kiểm chứng)

| config | thời gian/fold (T4×2) | dự đoán CV | vai trò |
|---|---|---|---|
| `tfidf` (CPU, local) | 1 phút | 0.85–0.92 | sàn CV, kiểm tra pipeline. **Không nộp** |
| `xlmr-base` | ~10 phút | 0.93–0.96 | **submission đầu tiên tối nay** |
| `xlmr-large` | ~30–40 phút | 0.96–0.98 | ứng viên chính |
| `mmbert-base` (max_len 1024) | ~15–25 phút | 0.95–0.98 | ứng viên chính |
| `deberta-pi-v2` | ~10 phút | ? (vocab thiên tiếng Anh) | thử nhanh |
| `granite-2b-lora` | ~2h | ? (2025: LLM thắng) | stretch, chỉ khi còn GPU |

## 2. Việc tối nay (21/10) — ai cũng làm

- [ ] **Join competition** trên Kaggle + accept rules (chưa join thì tải data bị 403).
- [ ] **Lập team trên Kaggle TRƯỚC KHI ai nộp bài** (gộp team sau khi đã nộp bị giới hạn số lượt). Tên team: `GenData`.
- [ ] Repo GitHub → **Private** (luật cấm chia sẻ code ngoài team trong lúc thi).
- [ ] Mỗi người: `make setup` → `make data` → `make eda` → `make baseline` (xem [WORKFLOW.md](WORKFLOW.md)).
- [ ] Mỗi người có `~/.kaggle/kaggle.json` của **chính mình**, kiểm tra quota GPU còn bao nhiêu giờ (Kaggle → Settings).
- [ ] Hỏi BTC trên Discussion: *"Được ensemble nhiều model khác nhau trong danh sách không, hay chỉ một?"*

### Chạy qua đêm (Kaggle chạy nền, không cần mở máy)

Cùng `seed: 42, n_folds: 5` → **cùng split** → fold 0 chạy tối nay **được dùng lại** trong bản cuối.

| người | lệnh | thời gian |
|---|---|---|
| A (lead) | `make push CONFIG=configs/xlmr-base.yaml` → nộp **submission #1** trước 07:00 sáng (vẫn tính quota ngày 21/10 UTC) | ~1h |
| B | `make push CONFIG=configs/xlmr-large.yaml FOLDS=0` | ~40 phút |
| C | `make push CONFIG=configs/mmbert-base.yaml FOLDS=0` | ~25 phút |
| D | `make push CONFIG=configs/deberta-pi-v2.yaml FOLDS=0` rồi `make push CONFIG=configs/granite-2b-lora.yaml` | ~3h |

## 3. Ngày thi (22/10) — timeline

| giờ VN | việc | ai |
|---|---|---|
| 08:00–08:30 | Đọc `report.md` fold 0 của các config (`make output KERNEL=...`). **Chốt backbone B\*** theo macro F1 tuned của fold 0 + per-language + thời gian/fold. Ghi vào `experiments/LOG.md` | cả team |
| 08:30–09:00 | Chạy **đủ 5 fold của B\*** chia cho 2–3 tài khoản: ví dụ B `FOLDS=1,2`, C `FOLDS=3,4` (fold 0 đã có). Đồng thời D chạy biến thể của B\* trên fold 0 (`prepend_language: true`, `max_len` dài hơn, `lr` khác) | B, C, D |
| ~10:30 | `make push-ensemble SOURCES=...` ghép 5 fold → **submission #2** (B\* 5-fold). So CV ↔ public LB | A |
| 10:30–13:30 | Phân tích lỗi trên OOF: confusion JAILBREAK↔PI, ngôn ngữ yếu nhất, prompt ngắn/dài. Biến thể nào thắng trên fold 0 → chạy đủ 5 fold (seed 2) | cả team |
| 13:30–15:30 | Ensemble 2 bộ 5-fold (2 seed / 2 config cùng backbone) → **submission #3–#4** | A |
| 15:30–16:30 | Build notebook chính thức: `tools/build_kernel.py ensemble --sources ... --final` → `RMIT_SGS_GenData_C1`. Nộp **submission cuối** | A |
| **17:00** | **Đóng băng.** Trên Kaggle chọn 2 *final submissions*: bản **CV cao nhất** + bản an toàn (không chọn theo public LB) | A |
| 18:00 | Deadline | |

**Quy tắc dùng 6 lượt nộp:** chỉ nộp khi CV (cross-fitted) tăng ≥ 0.003 hoặc để kiểm tra một giả thuyết cụ thể.
Giữ 1 lượt dự phòng đến 17:00. Mọi lượt nộp đi qua `tools/submit.py` (validate + log vào `experiments/submissions.csv`).

## 4. Phân vai

| | vai trò | phụ trách |
|---|---|---|
| A | Lead / integrator | repo, ensemble, quyết định nộp bài, notebook chính thức |
| B | Track XLM-R | `xlmr-large` (+ biến thể lr / max_len) |
| C | Track mmBERT | `mmbert-base` (+ max_len 1024 → 2048) |
| D | Phân tích lỗi + track phụ | `deberta-pi-v2`, `granite-2b-lora`, error analysis, slide / báo cáo |

## 5. Danh sách ý tưởng (ưu tiên từ trên xuống)

1. Offset theo lớp trên OOF (đã có) — luôn bật.
2. `class_weight`: `sqrt` vs `balanced` vs `none` (+ offset) — chạy fold 0.
3. `prepend_language: true` — cho model biết ngôn ngữ (đề gợi ý dùng cột `language`).
4. `max_len` dài hơn cho mmBERT (1024 → 2048) nếu EDA cho thấy nhiều prompt bị cắt (`truncated` trong report).
5. Thêm seed của B\* (ensemble cùng backbone, chắc chắn hợp lệ).
6. LLM `granite-2b-lora` nếu encoder dừng ở < 0.97 và còn ≥ 3h GPU.
7. Ensemble nhiều backbone — **chỉ khi BTC xác nhận được phép**.

## 6. Rủi ro & cách xử lý

| rủi ro | dấu hiệu | xử lý |
|---|---|---|
| XLM-R large "chết" (đoán toàn 1 lớp) | fold macro F1 ≈ 0.26, loss phẳng ~1.0 | `lr: 8.0e-6`, `warmup_ratio: 0.15`, đổi seed |
| fp16 NaN (mmBERT) | loss `nan` | `fp16: false` (chậm hơn ~2×) hoặc `lr` thấp hơn |
| Hết VRAM | CUDA OOM | `batch_size` ↓, `grad_accum` ↑ (giữ batch hiệu dụng), `grad_ckpt: true` |
| Kaggle hết 12h | notebook bị kill | artifact đã lưu sau mỗi fold; `time_budget_hours` dừng sớm |
| Hết quota GPU tài khoản | push bị từ chối | chuyển fold sang tài khoản khác (cùng config ⇒ cùng split) |
| Nộp sai format | mất 1 lượt | `tools/submit.py` validate trước khi nộp |
| Overfit public LB | public tăng, CV giảm | quyết định theo CV cross-fitted |
