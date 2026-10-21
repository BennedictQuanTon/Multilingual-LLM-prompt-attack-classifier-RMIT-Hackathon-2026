# Experiment log

Một dòng mỗi run. Điểm so sánh = **macro F1 (tuned, cross-fitted)** trong `report.md` (cột `CV`).
Split mặc định `seed 42 · 5 folds` — ghi fingerprint để chắc các run ghép được với nhau.

| # | ngày giờ | ai | config | folds | split | CV argmax | **CV tuned (x-fit)** | PI F1 | lang yếu nhất | phút/fold | kernel (owner/slug v#) | public LB | ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 21/10 20:54 | lead | tfidf | 0-4 | 70e83271a091 | 0.9271 | **0.9310** | 0.918 | pt (0.911; JB F1 0.85) | 0.25 | local | — | sàn CV, không nộp. Lỗi chính B↔J ở pt |
