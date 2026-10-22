# Experiment log

Một dòng mỗi run. Điểm so sánh = **macro F1 (tuned, cross-fitted)** trong `report.md` (cột `CV`).
Split mặc định `split_seed 42 · 5 folds` = **`10a79e50dde8`** (giống nhau trên local và Kaggle) — ghi fingerprint để chắc các run ghép được với nhau.

| # | ngày giờ | ai | config | folds | split | CV argmax | **CV tuned (x-fit)** | PI F1 | lang yếu nhất | phút/fold | kernel (owner/slug v#) | public LB | ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 21/10 21:30 | lead | tfidf | 0-4 | 10a79e50dde8 | 0.9304 | **0.9314** | 0.923 | pt (0.916; JB F1 0.855) | 0.25 | local | — | sàn CV, không nộp. Lỗi chính B↔J ở pt |
| 1 | 21/10 21:12 | lead | xlmr-large (2 GPU) | 0 | — | — | — | — | — | — | longquanton/rmit-sgs-gendata-xlmr-large-f0-c1 v1 | — | **CUDA OOM** GPU 0 ở backward đầu tiên → sửa: n_gpu 1, freeze_embeddings, grad_ckpt |
| 2 | 21/10 21:40 | lead | tfidf (fold 0, mốc benchmark) | 0 | 10a79e50dde8 | 0.9354 | 0.9378 (tuned in-sample) | — | pt | 0.2 | local | — | mốc GĐ1: model GPU phải vượt rõ |
| 3 | 21/10 21:37 | lead | tfidf **holdout 0.2** | 0-4 dev | c03bfc97563a | 0.9217 (holdout argmax) | **holdout 0.9253** (CI 0.912–0.937) · dev x-fit 0.9257 | 0.913 | pt (JB 0.836) | 0.2 | local | — | mốc GĐ2. CV dev ≈ holdout → CV đáng tin |
