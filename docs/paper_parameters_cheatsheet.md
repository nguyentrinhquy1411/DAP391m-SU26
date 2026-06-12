# 📄 AES-RARR Parameters Cheatsheet

Bảng tra cứu nhanh (Cheatsheet) này tổng hợp các tham số, biến số, và hệ số được sử dụng trong bài báo **AES-RARR** (Active Evidential Sensing for Maritime Search and Rescue) cùng với code mô phỏng tương ứng.

---

## 1. Dữ liệu đầu vào (System & Algorithmic Inputs)

| Ký hiệu | Tên gọi | Ý nghĩa |
| :--- | :--- | :--- |
| $\mathbf{I}_{RGB}, \mathbf{I}_{Thermal}$ | Sensory Feeds | Ảnh camera quang học (RGB) và ảnh nhiệt (Thermal). |
| $\mathbf{T}_{tel}$ | Telemetry Data | Dữ liệu viễn trắc của UAV, bao gồm vị trí ($\mathbf{p}_{UAV}$), độ cao ($z_{UAV}$), góc camera ($\boldsymbol{\theta}_{cam}$). |
| $\mathbf{b}_{i,t}$ | Bounding Box | Khung giới hạn (bounding box) phát hiện mục tiêu $i$ tại thời điểm $t$. |
| $\mathbf{x}_{i,t}$ | Projected Position | Vị trí mục tiêu $i$ chiếu xuống mặt biển. |
| $\mathbf{h}_{i,t}$ | Geometric Feature | Vector 7 chiều đại diện cho đặc trưng hình học (tỉ lệ, kích thước, khoảng cách đến tâm) đưa vào EDL. |

---

## 2. Mô hình Evidential Deep Learning (EDL)

| Ký hiệu | Code tương ứng | Tên gọi | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| $\mathbf{e}_{i,t}$ | `e` | Evidence Parameters | Bằng chứng (evidence) đầu ra từ mạng Softplus cho $K$ trạng thái (Drowning, Floating, Swimming, PFD). |
| $\alpha_{i,t,k}$ | `alpha` | Dirichlet Parameters | Tham số phân phối Dirichlet: $\alpha_k = e_k + 1$. |
| $S_{i,t}$ | `S` | Total Evidence | Tổng lượng bằng chứng: $S = \sum \alpha_k$. |
| $b_{i,t,k}$ | `beliefs` | Belief Mass | Xác suất tin tưởng (niềm tin) vào một class: $b_k = e_k / S$. |
| $u_{i,t}$ | `u` | Epistemic Uncertainty| Độ bất định (không chắc chắn do thiếu thông tin/bị sóng che khuất): $u = K / S$. Nằm trong khoảng $(0, 1]$. |
| $\mathbb{E}[R_i(t)]$ | `er` | Expected Risk | Rủi ro kỳ vọng của mục tiêu, tính bằng trung bình có trọng số của xác suất dự đoán và rủi ro sinh tồn. |

---

## 3. Theo dõi & Cảm biến chủ động (Tracking & Active Sensing)

| Ký hiệu | Code tương ứng | Tên gọi | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| $\boldsymbol{\Sigma}^{cam}, \boldsymbol{\Sigma}^{geo}$| `sc` | Spatial Covariance | Ma trận hiệp phương sai sai số không gian (trên ảnh và trên mặt biển). |
| $\gamma$ | `gamma_r` (2.0) | Covariance Scaler | Hệ số nhân độ phóng đại sai số đo lường dựa trên $u_{i,t}$. (Càng bất định, càng ít tin tưởng vào cảm biến hiện tại). |
| $\tau_{unc}$ | `tau_unc` (0.55) | Uncertainty Threshold| Ngưỡng độ bất định để kích hoạt nhánh bay thấp kiểm tra (Active branch). |
| $\tau_{risk}$ | `tau_risk` (0.30)| Risk Threshold | Ngưỡng rủi ro để kích hoạt nhánh kiểm tra. Cả 2 ngưỡng $u > \tau_{unc}$ và $\mathbb{E}[R] > \tau_{risk}$ phải thỏa mãn. |
| $z_{search}$ | `z_search` (40m)| Search Altitude | Độ cao bay tuần tra tiêu chuẩn. |
| $z_{verify}$ | `z_verify` (20m)| Verification Altitude| Độ cao bay thấp để xác minh mục tiêu khi bị che khuất. |
| $\rho$ | `rho` (0.5) | Altitude Ratio | Tỉ lệ giảm độ cao ($z_{verify} / z_{search}$). Làm giảm sai số đo lường $\rho^2$. |

---

## 4. Lập lịch Cứu hộ (Risk-Averse Routing)

| Ký hiệu | Code tương ứng | Tên gọi | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| $\lambda$ | `lambda_ra` (0.8) | Risk-Aversion Factor| Hệ số UCB (Upper Confidence Bound) kiểm soát mức độ ưu tiên thăm dò các mục tiêu có độ bất định cao. |
| $P^{rap}_i(t)$| `p_raw` | Risk-Averse Score | Điểm rủi ro kết hợp giữa Expected Risk và Uncertainty. |
| $\eta_i$ | `decay_rates` | Survival Decay Rate | Tốc độ suy giảm sinh lý/tỷ lệ sống sót theo thời gian. (Drowning: 0.08, Floating: 0.04, Swimming: 0.02, PFD: 0.01). |
| $D_i(t)$ | `dist` | Distance | Khoảng cách từ UAV đến mục tiêu. |
| $v_{agent}$ | `uav_speed` (10.0)| Agent Speed | Vận tốc bay của UAV (m/step). |
| $t_{travel,i}$| `tt` | Travel Time | Thời gian bay dự kiến tới mục tiêu. Nếu active branch kích hoạt, cộng thêm `descent_latency` (1.0). |

---

## 5. Các chỉ số đánh giá (Metrics)

| Ký hiệu | Tên gọi | Ý nghĩa |
| :--- | :--- | :--- |
| **VSR** | Victim Survival Rate | Tỷ lệ sinh tồn của nạn nhân $\vsr_i = \exp(-\eta_i \ttr_i)$. |
| **TTR** | Time to Rescue | Thời gian cần thiết (steps) để xác minh/cứu được nạn nhân (tính cả thời gian hạ độ cao). |
| **Worst VSR**| Worst-case Survival | Tỷ lệ sống sót của nạn nhân tồi tệ nhất trong 1 trial ($\min \vsr_i$). Đây là metric an toàn cốt lõi. |
| **CFR** | Catastrophic Failure Rate| Tỉ lệ thảm họa (số trials có Worst VSR < 0.35). |
| **Branch Precision** | Branch Precision | Độ chính xác của việc kích hoạt bay thấp (UAV có thực sự bay thấp cho mục tiêu nguy hiểm hay không). |

> [!TIP]
> **Điểm mấu chốt của Paper:** 
> Trong điều kiện sóng lớn (Wave Occlusion), mô hình thường cho ra *Expected Risk* không rõ ràng. Việc dùng $\tau_{unc}$ và $\gamma$ giúp UAV không bỏ qua mục tiêu đó (không bị xem là an toàn một cách chủ quan) mà sẽ kích hoạt cơ chế bay thấp ($z_{verify}$) để thu thập thêm bằng chứng (Evidence), trước khi bộ lập lịch chốt điểm ưu tiên cứu hộ cuối cùng.
