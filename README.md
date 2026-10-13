# Đề Tài 13: Hệ Thống Phát Hiện Gian Lận Giao Dịch Thẻ Tín Dụng (Credit Card Fraud Detection System)

> **Mô phỏng vận hành thực tế tại ngân hàng (Real-world Bank Simulation)**: Hệ thống nạp file giao dịch thô chưa có nhãn (*Unlabeled CSV*), tự động trích xuất 11 đặc trưng phi rò rỉ, chuẩn hóa `RobustScaler`, suy luận qua mô hình **Master Model LightGBM (297 cây)**, và ra quyết định chặn/duyệt theo ngưỡng tối ưu kinh tế $T = 0.0227$.

---

## 📊 Bảng Chỉ Số Nghiệm Thu Thực Tế (Master Model Benchmark)

Toàn bộ các chỉ số được đo lường trên tập kiểm thử độc lập (**Test Set: 15.000 giao dịch**):

| Thước đo | Giá trị nghiệm thu | Ý nghĩa nghiệp vụ |
| :--- | :---: | :--- |
| **PR-AUC (Test)** | **0.7699** | Đo trên xác suất liên tục `predict_proba[:, 1]` |
| **Recall (Test)** | **85.45%** | Bắt trúng **47 / 55 vụ** gian lận (chỉ lọt lưới 8 vụ nhỏ) |
| **Precision (Test)** | **17.87%** | Base rate: 0.37% (Hiệu quả gấp **~48 lần** ngẫu nhiên) |
| **F1-Score** | **0.2956** | Điểm cân bằng tại vùng dữ liệu mất cân bằng cực đoan |
| **Ngưỡng quyết định ($T$)** | **0.0227** (2.27%) | Dò tìm bằng phương pháp Threshold Moving trên tập Validation |
| **Thiệt hại do bỏ sót (FN)** | **$928.24** | 8 vụ gian lận lọt lưới |
| **Chi phí cảnh báo nhầm (FP)** | **$1,080.00** | 216 cuộc gọi xác minh CSKH ($5 / cuộc) |
| **Tổng thiệt hại thực tế** | **$2,008.24** | $\text{Loss} = \sum (\text{FN} \times \text{amt}) + \sum (\text{FP} \times 5\$)$ |
| **Số tiền gian lận gốc** | **$28,783.63** | Thiệt hại ngân hàng chịu nếu không có mô hình AI |
| **Số tiền tiết kiệm được** | **$26,775.39** | **Giảm 93.02% tổng thiệt hại tài chính** |
| **Thời gian huấn luyện** | **0.268 giây** | 40.000 dòng Train (297 cây) |
| **Độ trễ suy luận đơn lẻ** | **0.8894 ms / tx** | P99: **1.78 ms** (Đạt chuẩn POS quốc tế < 50 ms) |

---

## 🏗️ Kiến Trúc Hệ Thống (System Architecture)

```
[Tệp CSV Giao dịch Thô (21 cột - KHÔNG CÓ CỘT is_fraud)]
                           │
                           ▼
          [1. Feature Engineering Engine]
   • Khoảng cách Haversine (km) giữa khách hàng và máy POS
   • Tuổi, chu kỳ giờ sin/cos, cờ đêm khuya (00:00 - 05:00)
   • Tỷ lệ chi tiêu đột biến (amt_ratio, amt_diff)
   • Tần suất giao dịch quá khứ (trans_count_past)
   • Tra cứu rủi ro ngành hàng (category_risk)
                           │
                           ▼
          [2. RobustScaler Transformation]
   • Chuẩn hóa dữ liệu theo trung vị (Median) và khoảng tứ phân vị (IQR)
                           │
                           ▼
          [3. LightGBM Master Model (297 Cây)]
   • predict_proba(X)[:, 1] ➔ Xác suất rủi ro r (0.00% - 100.00%)
                           │
                           ▼
          [4. Business Decision Policy Engine]
   • r < 2.27%          ➔ AUTO-APPROVED (Tự động duyệt - Xanh)
   • 2.27% ≤ r ≤ 15.0%  ➔ STEP-UP AUTH / CSKH (Cần gọi xác minh / OTP - Vàng)
   • r > 15.0%          ➔ AUTO-BLOCKED (Chặn tức thì / Khóa thẻ - Đỏ)
                           │
                           ▼
          [5. Sentinel Fraud Operations Web App]
   • Giám sát luồng giao dịch trực tiếp (Live Feed)
   • Điều tra chuyên sâu từng vụ án (Explainable AI - XAI)
   • Dashboard hiệu năng & Báo cáo tác động tài chính
```

---

## 📂 Cấu Trúc Thư Mục Dự Án

```
fraud-operations-web-demo/
│
├── backend/
│   ├── models/
│   │   ├── master_model.joblib       # Trọng số mô hình LightGBM đã train (297 cây)
│   │   ├── master_model.txt          # Định dạng cây LightGBM gốc
│   │   ├── scaler.joblib             # RobustScaler đã fit trên 40k dòng
│   │   ├── meta_features.joblib      # Metadata, từ điển category_risk và card profiles
│   │   └── meta_features.json        # Thông số cấu hình dạng JSON
│   ├── app.py                        # Flask Web Server & Real-time Scoring API
│   ├── inference_engine.py           # Động cơ trích xuất 11 features & chấm điểm AI
│   └── train_and_export.py           # Script huấn luyện & xuất model từ dataset
│
├── fraud-portal/
│   ├── index.html                    # Giao diện Web Sentinel Fraud Operations Portal
│   ├── demo_transactions.csv        # File giao dịch mẫu ban đầu
│   ├── unlabeled_transactions_demo.csv # 100 giao dịch test THỰC TẾ (ĐÃ BỎ NHÃN is_fraud)
│   └── scored_demo_transactions.json # Kết quả chấm điểm sẵn
│
├── requirements.txt                  # Các thư viện Python cần thiết
├── .gitignore                        # Cấu hình bỏ qua file rác và dataset lớn
└── README.md                         # Tài liệu hướng dẫn đồ án
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### 1. Cài đặt môi trường Python
Yêu cầu Python >= 3.9:
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy hệ thống Web
Chạy server backend Flask:
```bash
python backend/app.py
```

Truy cập giao diện tại trình duyệt:
👉 **`http://127.0.0.1:5000`**

---

## 🎯 Quy Trình Demo Nghiệp Vụ Thực Tế

1. **Nạp dữ liệu không có nhãn**:
   * Bấm nút **"📥 Tải tệp mẫu không nhãn"** để lấy tệp `unlabeled_transactions_demo.csv` (gồm 100 giao dịch thực nghiệm từ tập test, **hoàn toàn không có cột `is_fraud`**).
   * Bấm nút **"📂 Nạp tệp CSV không nhãn"** và chọn tệp vừa tải.
2. **AI Chấm điểm tức thì**:
   * Mô hình xử lý toàn bộ 100 giao dịch trong **~68 ms** (0.68 ms/tx).
   * Phân luồng tự động: **81 ca duyệt**, **3 ca cần CSKH gọi**, **16 ca chặn gian lận**.
3. **Điều tra chuyên sâu vụ án (Explainable AI)**:
   * Bấm vào bất kỳ dòng nào để xem hồ sơ chủ thẻ, vị trí toạ độ POS, khoảng cách di chuyển và các thẻ **Explainable AI** giải thích rõ nguyên nhân bị gắn cờ (số tiền đột biến, giao dịch lúc 2h sáng, lệch toạ độ xa).
4. **Xuất báo cáo kiểm toán**:
   * Bấm **"⇩ Xuất kết quả đã chấm điểm"** để tải file CSV hoàn chỉnh có kèm điểm rủi ro và mã hành động của AI.
