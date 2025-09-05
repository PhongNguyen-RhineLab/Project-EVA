# Project EVA

**Empathic Voice Assistant (EVA) to help people with psychological disorder**

*Trợ lý ảo Thấu cảm sử dụng Mô hình Ngôn ngữ Lớn và Phân tích Giọng nói Đa nhãn*

---

## 1. Mục tiêu chính

Xây dựng một hệ thống AI hoàn chỉnh có khả năng:

1. Hiểu nội dung văn bản của người nói qua việc chuyển đổi giọng nói thành văn bản (Speech-to-Text).
2. Cảm nhận trạng thái cảm xúc phức hợp của người nói bằng cách phân tích các đặc tính âm học của giọng nói, cho ra một vector xác suất đa nhãn.

   * Ví dụ: **Vui 30%, Buồn 70%**
3. Phản hồi một cách thông minh và thấu cảm bằng cách sử dụng một **mô hình ngôn ngữ lớn (LLM)** kết hợp với thông tin về cả nội dung và cảm xúc.

---

## 2. Kiến trúc Hệ thống Tổng thể

**Luồng xử lý dữ liệu:**

```
Giọng nói người dùng
   ├─> (1) Speech-to-Text (STT) → Văn bản nội dung
   ├─> (2) Nhận diện Cảm xúc (SER) → Vector cảm xúc
   └─> Bộ tạo Prompt Thích ứng → LLM → Văn bản phản hồi → Text-to-Speech (TTS)
```

---

## 3. Chi tiết Kỹ thuật các Module chính

### Module 1: Thu thập và Tiền xử lý Dữ liệu

* **Nguồn dữ liệu**

  * Công khai (tiếng Anh): RAVDESS, IEMOCAP, Emo-DB.
  * Dài hạn: Bộ dữ liệu tiếng Việt gán nhãn cảm xúc.

* **Tiền xử lý**

  * Biểu diễn: Chuyển audio thành **Mel Spectrogram**.
  * Tham số đề xuất: `n_mels=128`, `hop_length=512`, `n_fft=2048`.
  * Chuẩn hóa spectrogram về `[0, 1]` hoặc `[-1, 1]`.

* **Gán nhãn (Labeling)**

  * Phân loại đa nhãn: Ví dụ, \[Vui, Buồn, Tức giận, Trung tính].
  * Ví dụ:

    * Audio **Buồn** → `[0, 1, 0, 0]`
    * Audio **Vui + Ngạc nhiên** → `[1, 0, 0, 0, 1]`

* **Bộ 8 cảm xúc chính**

  * Trung tính (Neutral)
  * Bình tĩnh (Calm)
  * Vui vẻ (Happy)
  * Buồn bã (Sad)
  * Tức giận (Angry)
  * Sợ hãi (Fearful)
  * Ghê tởm (Disgust)
  * Ngạc nhiên (Surprised)

> ⚠️ **Hạn chế:** Không dễ nắm bắt cảm xúc xã hội bậc cao như Tội lỗi, Xấu hổ, Tự hào, Ghen tị.

---

### Module 2: Mô hình SER (Speech Emotion Recognition)

* **Mô hình**: Multi-task Beta-VAE

* **Encoder**

  * CNN + LSTM/GRU
  * Conv2D + BatchNorm + MaxPooling
  * LSTM/GRU để học phụ thuộc thời gian

* **Latent Space**

  * Tạo `μ` (mean vector) và `log_var` (variance log).
  * Sampling vector `z` (32 → 128 dims).

* **Decoder**

  * Tái tạo spectrogram từ `z`.

* **Classifier Head**

  * Dense + Dropout → Output sigmoid (đa nhãn).

* **Loss Function**

L_total = α * L_classification + β * L_kld + γ * L_reconstruction
L_classification = BCE(y_true, y_pred)
L_reconstruction = MSE(x_true, x_reconstructed)
L_kld = -0.5 * Σ(1 + log_var - μ² - exp(log_var))

---

### Module 3: Tích hợp LLM (Gemma)

**Bộ tạo Prompt Thích ứng (Context-Adaptive Prompt Engine)**

**Template Prompt ví dụ:**

```
Bối cảnh hệ thống: Bạn là một trợ lý ảo thấu cảm.  
Phân tích cảm xúc từ giọng nói: Người dùng có vẻ buồn bã (70%) và mệt mỏi (40%).  
Nhiệm vụ: Hãy phản hồi an ủi, tránh lời khuyên trực tiếp trừ khi được hỏi.  
Nội dung người dùng: [Văn bản STT]
```

---

## 4. Roadmap (Lộ trình Triển khai)

* **Giai đoạn 1:** Huấn luyện SER
* **Giai đoạn 2:** Prototype End-to-End
* **Giai đoạn 3:** Tinh chỉnh & Đánh giá

---

## 5. Thách thức Chính

* **Dữ liệu:** Thiếu dataset tiếng Việt gán nhãn.
* **Cân bằng Loss:** Điều chỉnh `α, β, γ` phức tạp.
* **Độ trễ:** Nhiều bước gây delay.
* **Đánh giá:** Thấu cảm khó đo lường bằng số.

---

## 6. Công nghệ Sử dụng

* **STT:** Whisper, Vosk, Google Speech, Azure Speech
* **SER:** Librosa, NumPy, PyTorch, CNN + LSTM
* **LLM:** Gemma, LLaMA, GPT-4, DeepSeek
* **TTS:** (Chưa quyết định)
* **Triển khai:** Python, Docker, REST API, React Native
