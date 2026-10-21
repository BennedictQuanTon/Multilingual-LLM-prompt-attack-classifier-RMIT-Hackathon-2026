# Plan — Team GenData · RMIT Hackathon 2026 · Challenge 1

> Cập nhật: **21/10/2026, 21:00 giờ VN**. Deadline Kaggle: **22/10/2026 18:00 giờ VN** (11:00 UTC).
> Mục tiêu: **CV macro F1 ≥ 0.975** và nộp bản cuối trước **17:00 ngày 22/10**.

---

## 1. Tình hình hiện tại

| hạng mục | trạng thái |
|---|---|
| Đăng ký cuộc thi (tài khoản lead `longquanton`) | ✅ đã Join |
| Repo GitHub | ✅ Private · CI chạy mỗi push |
| Dữ liệu | ✅ đã tải (`make data`) · EDA xong (`reports/eda.md`) |
| Baseline TF-IDF (CPU, local) | ✅ **CV 0.9310** (cross-fitted) — chỉ để làm mốc, **không nộp** |
| Chạy GPU đầu tiên trên Kaggle | ⏳ `longquanton/rmit-sgs-gendata-xlmr-large-f0-c1` (XLM-R large, fold 0) |
| Submission | 0 / 6 (quota ngày 21/10 UTC còn tới **07:00 sáng 22/10 giờ VN**) |
| Team trên Kaggle | ☐ cần lập (tối đa 4 người) **trước khi ai nộp bài** |
| Câu hỏi gửi BTC | ☐ "Được ensemble nhiều model trong danh sách không, hay chỉ một?" |

**Leaderboard public (21/10 ~20:00):** top **0.97185** · hạng 5 ≈ 0.961 · notebook mẫu của BTC 0.908 · đoán toàn BENIGN ≈ 0.26.
Public LB chỉ chấm **30% test (~1.800 dòng)** → nhiễu ±0.01. Thứ hạng cuối dựa trên 70% còn lại (private).

## 2. Luật — những điều bắt buộc

| # | luật | hệ quả cho mình |
|---|---|---|
| 1 | Chỉ dùng **MỘT backbone** trong 10 model được duyệt; ghi tên model ở đầu notebook | `build_kernel.py` tự ghi; `ensemble` chặn trộn backbone (trừ khi BTC cho phép) |
| 2 | **6 lượt nộp / ngày**, reset 07:00 giờ VN | mỗi lượt đi qua `tools/submit.py` (validate + log) |
| 3 | Team tối đa **4 người**, mỗi người 1 tài khoản Kaggle | lập team **trước** khi nộp — gộp sau bị giới hạn lượt |
| 4 | Tên notebook `<university>_<campus>_<team>_C1` | bản chính thức: **`RMIT_SGS_GenData_C1`**; thí nghiệm: `RMIT_SGS_GenData-<exp>_C1` (vẫn đúng luật) |
| 5 | "Submit notebooks to get higher score" | nộp từ **output của notebook**, không nộp file làm tay |
| 6 | Có điểm C1 ngay khi có **submission hợp lệ đầu tiên** | nộp sớm 1 bản tối nay |
| 7 | Không chia sẻ code ngoài team trong lúc thi; dữ liệu CC BY-NC-SA | repo Private, không commit data (pre-commit + CI chặn) |
| 8 | Không dùng nhãn tay / dự đoán của người trên test | pseudo-label bằng model thì được, gán tay thì không |

Danh sách model được duyệt (key dùng trong `configs/`):

| key | model | loại |
|---|---|---|
| `xlmr-base` / `xlmr-large` | FacebookAI/xlm-roberta-base / large | encoder đa ngữ |
| `mmbert-small` / `mmbert-base` | jhu-clsp/mmBERT-small / base | encoder đa ngữ, context dài |
| `distilbert-mbert` | distilbert-base-multilingual-cased | encoder nhỏ |
| `rembert` | google/rembert | encoder (batch 8) |
| `deberta-pi-v2` | protectai/deberta-v3-base-prompt-injection-v2 | encoder đã học prompt injection (EN) |
| `wolf-defender` | patronus-studio/wolf-defender-prompt-injection | encoder đã học prompt injection |
| `granite-2b` | ibm-granite/granite-3.3-2b-instruct | LLM (4-bit + LoRA) |
| `mistral-7b` | mistralai/Mistral-7B-Instruct-v0.3 | LLM (4-bit + LoRA) |

## 3. Đề bài & dữ liệu — những gì đã biết chắc

- 3 lớp: `BENIGN` 62% · `JAILBREAK` 31% · `PROMPT_INJECTION` 7%; 4 ngôn ngữ: pt 45% · fr 25% · it 20% · en 10%.
- **Mọi ngôn ngữ có đúng tỉ lệ 62/31/7**, không có prompt trùng, test không trùng train, phân bố train = test
  → không có leak để khai thác; thắng bằng model.
- Prompt ngắn: **median 71 token, trung bình 120**; `max_len 512` chỉ cắt 2.1%.
- **Macro F1** → lớp PI (7%) nặng bằng BENIGN (62%) → phải chỉnh ngưỡng theo lớp (đã có, tự bật).
- **Điểm nghẽn #1 — BENIGN ↔ JAILBREAK trong tiếng Bồ Đào Nha:** TF-IDF sai ~11% ở pt, chỉ 1–2.5% ở fr/it/en
  (~610/820 lỗi B↔J). Đó là các prompt **nhập vai dài** mà *cả hai lớp đều nhập vai*; khác nhau ở chỗ yêu cầu bên dưới
  có hại hay không → cần model **hiểu nghĩa**, không phải bắt từ khoá.
- **Điểm nghẽn #2 — PROMPT_INJECTION bị đoán BENIGN:** 112 lỗi, tệ nhất ở **en (sai 24.5%)**: injection ngắn, mập mờ
  ("Sans tenir compte des articles…"), có thể có nhãn nhiễu.

→ **Chọn model theo `f1_JAILBREAK` của pt và `f1_PROMPT_INJECTION` của en** (bảng *Per language* trong `report.md`),
không chỉ theo macro F1 tổng.

## 4. Ta đang có gì

| công cụ | dùng để |
|---|---|
| `configs/*.yaml` | 1 file = 1 thí nghiệm: `tfidf`, `xlmr-base`, `xlmr-large`, `mmbert-base`, `deberta-pi-v2`, `granite-2b-lora`, `mistral-7b-lora` |
| `make push CONFIG=… [FOLDS=…]` | build notebook từ code hiện tại → đẩy lên Kaggle → chạy trên T4×2 |
| `make status / output KERNEL=…` | xem trạng thái / tải kết quả về `outputs/<slug>/` |
| `artifacts/<name>/report.md` | **CV cross-fitted**, per class, per language, confusion, thời gian, VRAM |
| `make push-ensemble SOURCES=…` | ghép fold / seed / config của **cùng backbone** từ notebook của nhiều người |
| `make submit-dry / submit KERNEL=…` | validate + đếm lượt hôm nay + nộp + ghi `experiments/submissions.csv` |
| `experiments/LOG.md` | nhật ký mọi run (ai, config, CV, ghi chú) |

**Split cố định:** `split_seed 42 · 5 folds` → fingerprint **`70e83271a091`**. Mọi run của mọi người dùng chung split này
⇒ fold chạy ở tài khoản nào cũng ghép được. Muốn thêm thành viên ensemble thì đổi `seed` (seed train), **không đổi `split_seed`**.

## 5. Chiến lược

1. **Benchmark tối nay:** mỗi model một notebook, **chỉ fold 0**, song song trên 4 tài khoản (không dồn vào một notebook chạy lần lượt).
   So *cặp* trên cùng fold 0; chênh **≥ 0.01** mới đáng tin (TF-IDF dao động 0.920–0.934 giữa các fold).
2. **Chốt 1 backbone (B\*) lúc 8:30 sáng mai** theo: CV fold 0 → pt JAILBREAK F1 → en PI F1 → thời gian/fold.
   Năm 2025 LLM (Qwen 3B) thắng encoder đúng ở kiểu lỗi "nhập vai" này → Granite / Mistral đáng thử dù chậm.
3. **Chạy đủ 5 fold của B\*** (fold 0 đã có từ benchmark), chia fold cho nhiều tài khoản rồi ghép.
4. **Tăng điểm trong luật:** thêm seed của B\* → biến thể config (nhãn ngôn ngữ, class weight) → pseudo-label test (nếu còn giờ).
5. **Chọn bản cuối theo CV cross-fitted**, không theo public LB.

## 6. TỐI NAY (21/10) — làm gì, chạy gì

### Ai cũng làm (15 phút)

```bash
git clone https://github.com/BennedictQuanTon/Team-GenData-RMIT-Hackathon-2026.git && cd Team-GenData-RMIT-Hackathon-2026
make setup                      # cài môi trường + pre-commit
# Kaggle → Settings → Create New Token → lưu ~/.kaggle/kaggle.json → chmod 600 ~/.kaggle/kaggle.json
# Kaggle web: Join competition + accept rules, chấp nhận lời mời vào team GenData
make data && make baseline      # phải ra CV 0.9310 giống lead → môi trường đúng
```

### Benchmark fold 0 (chạy nền trên Kaggle, không cần mở máy)

| người | lệnh | thời gian ước tính | ghi chú |
|---|---|---|---|
| **A (lead)** | *(đã chạy)* `make push CONFIG=configs/xlmr-large.yaml FOLDS=0` | ~20 phút | xong → **nộp submission #1** (mục 8) |
| **B** | `make push CONFIG=configs/mmbert-base.yaml FOLDS=0` → xong thì `make push CONFIG=configs/xlmr-base.yaml FOLDS=0` | ~15 + 10 phút | |
| **C** | `make push CONFIG=configs/granite-2b-lora.yaml` | ~1–1.5 giờ | config đã đặt sẵn fold 0 |
| **D** | `make push CONFIG=configs/mistral-7b-lora.yaml` → xong thì `make push CONFIG=configs/deberta-pi-v2.yaml FOLDS=0` | ~2.5–3 giờ + 10 phút | để chạy qua đêm |

Sau khi push: mở link Kaggle in ra → tab *Logs* xem 5 phút đầu có lỗi không.
Mỗi người **Share notebook cho lead** (*Share → Add collaborator*) để sáng mai ghép được.

Trước khi ngủ: `make output KERNEL=<user>/<slug>` → đọc `outputs/<slug>/artifacts/<name>/report.md` → ghi 1 dòng vào `experiments/LOG.md`.

## 7. NGÀY THI (22/10) — timeline

Ví dụ dưới đây giả sử B\* = `xlmr-large`; nếu chọn model khác thì thay tên config.
Slug notebook có dạng `rmit-sgs-gendata-<config>-f<các fold>-c1`.

| giờ VN | việc | ai | lệnh |
|---|---|---|---|
| **07:00** | Quota nộp bài mới (6 lượt) | | |
| 08:00–08:30 | Tải kết quả benchmark, điền `LOG.md`, **chốt B\*** | cả team | `make output KERNEL=<user>/<slug>` |
| 08:30 | Chạy 4 fold còn lại của B\* trên 2 tài khoản | B, C | B: `make push CONFIG=configs/xlmr-large.yaml FOLDS=1,2` · C: `make push CONFIG=configs/xlmr-large.yaml FOLDS=3,4` |
| 08:30 | Biến thể của B\* trên fold 0 | D | tạo `configs/xlmr-large-lang.yaml` (copy, đổi `name`, thêm `prepend_language: true`) → `make push CONFIG=configs/xlmr-large-lang.yaml FOLDS=0` |
| 08:30–10:00 | Phân tích lỗi OOF của B\* fold 0: pt B↔J, en PI | A | đọc `report.md`; `oof.npy` + `data/train.csv` |
| ~10:00 | **Ghép 5 fold → submission #2** | A | `make push-ensemble SOURCES=longquanton/rmit-sgs-gendata-xlmr-large-f0-c1,<userB>/rmit-sgs-gendata-xlmr-large-f12-c1,<userC>/rmit-sgs-gendata-xlmr-large-f34-c1` → `make submit KERNEL=longquanton/rmit-sgs-gendata-ensemble-c1 MSG="B* 5f cv=…"` |
| 10:00–13:00 | Seed thứ 2 của B\* (5 fold, chia 2–3 tài khoản) + biến thể thắng ở fold 0 chạy đủ 5 fold | B, C, D | copy config → đổi `name: xlmr-large-s2`, `seed: 7` → `make push CONFIG=configs/xlmr-large-s2.yaml FOLDS=…` |
| 13:00–14:00 | Ghép (B\* + seed 2 + biến thể) → **submission #3** nếu CV tăng ≥ 0.003 | A | `make push-ensemble SOURCES=<tất cả notebook của B*>` |
| 14:00–15:30 | Nếu còn GPU: pseudo-label test / seed 3. Không thì chuẩn bị slide / báo cáo | cả team | |
| **15:30–16:30** | **Notebook chính thức** `RMIT_SGS_GenData_C1` = ensemble tốt nhất → **submission cuối** | A | `uv run python tools/build_kernel.py ensemble --sources <…> --final` → `uv run kaggle kernels push -p kaggle/build/rmit-sgs-gendata-c1` → `make submit KERNEL=longquanton/rmit-sgs-gendata-c1 MSG="final cv=…"` |
| **17:00** | **Đóng băng.** Kaggle → *Submissions* → chọn 2 bản cuối: **CV cao nhất** + 1 bản an toàn | A | |
| 18:00 | Deadline | | |

Nếu B\* là LLM (Granite ~1.5h/fold, Mistral ~3h/fold, 1 GPU): chia 4 fold còn lại cho **4 tài khoản**
(`FOLDS=1`, `FOLDS=2`, `FOLDS=3`, `FOLDS=4`) ngay 08:30, bỏ seed 2, ghép lúc ~10:30 (Granite) hoặc ~12:00 (Mistral).

## 8. Quy tắc nộp bài

- Chỉ nộp khi CV cross-fitted tăng **≥ 0.003**, hoặc để kiểm tra một giả thuyết cụ thể. Giữ **1 lượt dự phòng** tới 17:00.
- Luôn chạy thử trước: `make submit-dry KERNEL=<user>/<slug> MSG="…"` (validate file, đếm lượt đã dùng hôm nay).
- Nộp thật: `make submit KERNEL=<user>/<slug> VERSION=<n> MSG="<config> <folds> cv=<…>"`.
  Nếu CLI báo lỗi khi nộp theo kernel: `make output KERNEL=…` rồi
  `uv run python tools/submit.py --file outputs/<slug>/submission.csv -m "…" --yes`, hoặc bấm *Submit* trong tab *Output* của notebook.
- Có điểm public → điền cột `public_lb` trong `experiments/submissions.csv` và `experiments/LOG.md`.
- **Tối nay:** submission #1 = output của `longquanton/rmit-sgs-gendata-xlmr-large-f0-c1` (1 fold, đủ để chắc điểm C1 và so CV ↔ LB).

## 9. Phân vai

| | vai trò | phụ trách |
|---|---|---|
| A | Lead / integrator (`longquanton`) | repo, chốt model, ensemble, mọi lượt nộp, notebook chính thức |
| B | Track encoder | `mmbert-base`, `xlmr-base`, sau đó fold của B\* |
| C | Track LLM nhỏ | `granite-2b-lora`, sau đó fold của B\* |
| D | Track LLM lớn + biến thể | `mistral-7b-lora`, `deberta-pi-v2`, biến thể config, slide / báo cáo |

## 10. Rủi ro & cách xử lý

| rủi ro | dấu hiệu | xử lý |
|---|---|---|
| XLM-R large không học (đoán toàn 1 lớp) | fold macro F1 ≈ 0.26, loss phẳng ~1.0 | `lr: 8.0e-6`, `warmup_ratio: 0.15`, đổi `seed` |
| fp16 ra NaN (mmBERT) | loss `nan` | `fp16: false` hoặc giảm `lr` |
| Hết VRAM | CUDA OOM | giảm `batch_size`, tăng `grad_accum`, `grad_ckpt: true` |
| Notebook bị kill (12 giờ) | dừng giữa chừng | artifact đã lưu sau mỗi fold; `time_budget_hours` tự dừng sớm |
| Hết quota GPU của một tài khoản | push bị từ chối | chuyển fold sang tài khoản khác (cùng config ⇒ cùng split) |
| Không ghép được notebook người khác | ensemble không thấy input | notebook nguồn phải Share cho lead; hoặc `make output` từng cái rồi `make ensemble SOURCES="outputs/a outputs/b"` |
| Nộp sai format | mất 1 lượt | `make submit-dry` trước |
| Bám public LB | public tăng, CV giảm | quyết định theo CV cross-fitted |

## 11. Lệnh tra nhanh

```bash
make help                                        # mọi lệnh
make check                                       # lint + test (giống CI)
make push CONFIG=configs/<x>.yaml FOLDS=0        # chạy trên Kaggle
make status KERNEL=<user>/<slug>                 # queued / running / complete / error
make output KERNEL=<user>/<slug>                 # tải kết quả
make push-ensemble SOURCES=<user>/<slug>,<user>/<slug>
make submit-dry KERNEL=<user>/<slug> MSG="…"     # kiểm tra trước khi nộp
make submit KERNEL=<user>/<slug> VERSION=<n> MSG="…"
```

Chi tiết cách vận hành: [docs/WORKFLOW.md](docs/WORKFLOW.md) · Đề bài đầy đủ: [docs/PROBLEM.md](docs/PROBLEM.md).
