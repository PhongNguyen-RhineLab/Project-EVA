<div align="center">

# 🎭 Project EVA

**Empathic Voice Assistant** - Trợ lý ảo Thấu cảm sử dụng Mô hình Ngôn ngữ Lớn và Phân tích Giọng nói Đa nhãn

![EVA Banner](https://github.com/user-attachments/assets/4a389759-37be-4c2f-a75c-e4b4e510dcc2)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Research-green.svg)]()
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()

</div>

## 📋 Tổng quan

Project EVA là một hệ thống AI tiên tiến được thiết kế để hỗ trợ người có rối loạn tâm lý thông qua việc phân tích cảm xúc từ giọng nói và cung cấp phản hồi thấu cảm, phù hợp với trạng thái tâm lý của người dùng.

### ✨ Tính năng chính

Hệ thống EVA kết hợp ba khả năng cốt lõi:

1. **🎤 Speech-to-Text (STT)** - Chuyển đổi giọng nói thành văn bản để hiểu nội dung câu chuyện
2. **😊 Speech Emotion Recognition (SER)** - Phân tích cảm xúc phức hợp qua đặc tính âm học (ví dụ: Vui 30%, Buồn 70%)
3. **🤖 LLM Integration** - Tạo phản hồi thấu cảm dựa trên cả nội dung và trạng thái cảm xúc
4. **🔊 Text-to-Speech (TTS)** - Phản hồi bằng giọng nói tự nhiên với hỗ trợ tiếng Việt (ElevenLabs, Edge TTS)

---

## 🏗️ Kiến trúc Hệ thống

```
┌─────────────────────┐
│  Giọng nói người    │
│      dùng           │
└──────────┬──────────┘
           │
           ├──────────────────────┐
           │                      │
           ▼                      ▼
    ┌──────────────┐      ┌──────────────┐
    │   STT        │      │     SER      │
    │  (Whisper)   │      │  (Beta-VAE)  │
    └──────┬───────┘      └──────┬───────┘
           │                     │
           │  Văn bản            │  Vector cảm xúc
           │                     │  [0.3, 0.7, ...]
           └─────────┬───────────┘
                     │
                     ▼
           ┌─────────────────────┐
           │  Prompt Engine      │
           │  (Context-Aware)    │
           └─────────┬───────────┘
                     │
                     ▼
           ┌─────────────────────┐
           │   LLM (Gemma)       │
           │   Tạo phản hồi      │
           └─────────┬───────────┘
                     │
                     ▼
           ┌─────────────────────┐
           │   TTS               │
           │   Phản hồi bằng     │
           │   giọng nói         │
           └─────────────────────┘
```

---

## 🎯 Module SER - Speech Emotion Recognition

### Mô hình: Multi-task Beta-VAE

#### 🔧 Kiến trúc

**Encoder:**
- CNN (Conv2D → BatchNorm → MaxPooling)
- LSTM/GRU cho phụ thuộc thời gian
- Tạo latent space: `μ` (mean) và `log_var` (variance)

**Latent Space:**
- Chiều: 32-128 dimensions
- Sampling: `z = μ + σ * ε` (reparameterization trick)

**Decoder:**
- Tái tạo spectrogram từ latent vector `z`

**Classifier Head:**
- Dense layers + Dropout
- Output: Sigmoid activation (multi-label)
- 8 cảm xúc đồng thời

#### 📊 8 Cảm xúc cơ bản

| Cảm xúc | Tiếng Anh | Mô tả |
|---------|-----------|-------|
| 😐 | Neutral | Trung tính |
| 😌 | Calm | Bình tĩnh |
| 😊 | Happy | Vui vẻ |
| 😢 | Sad | Buồn bã |
| 😠 | Angry | Tức giận |
| 😨 | Fearful | Sợ hãi |
| 🤢 | Disgust | Ghê tởm |
| 😲 | Surprised | Ngạc nhiên |

> ⚠️ **Lưu ý:** Mô hình hiện tại không nắm bắt tốt các cảm xúc xã hội phức tạp như tội lỗi, xấu hổ, tự hào, ghen tị.

#### 🔬 Loss Function

```python
L_total = α·L_classification + β·L_kld + γ·L_reconstruction

Trong đó:
• L_classification = BCE(y_true, y_pred)
• L_reconstruction = MSE(x_true, x_reconstructed)  
• L_kld = -0.5·Σ(1 + log_var - μ² - exp(log_var))
```

**Hyperparameters:**
- `α`: Trọng số classification (mặc định: 1.0)
- `β`: Trọng số KL divergence (mặc định: 0.5, có warmup)
- `γ`: Trọng số reconstruction (mặc định: 0.1)

---

## 🔌 Module LLM - Context-Aware Prompt

### Prompt Template

```
Bối cảnh hệ thống: 
Bạn là EVA - trợ lý ảo thấu cảm được thiết kế để hỗ trợ 
người có rối loạn tâm lý.

Phân tích cảm xúc từ giọng nói:
Người dùng đang cảm thấy: Buồn bã (70%), Mệt mỏi (40%)

Hướng dẫn phản hồi:
• Thể hiện sự đồng cảm và thấu hiểu
• Công nhận trạng thái cảm xúc một cách tự nhiên
• Tránh đưa ra lời khuyên trực tiếp trừ khi được hỏi
• Sử dụng giọng điệu ấm áp, thân thiện
• Giữ phản hồi ngắn gọn nhưng ý nghĩa

Nội dung người dùng: [Văn bản từ STT]

Phản hồi thấu cảm của bạn:
```

---

## 🗂️ Cấu trúc Dự án

```
Project-EVA/
├── 📁 VAE/                      # Mô hình Speech Emotion Recognition
│   ├── model.py                 # Kiến trúc Beta-VAE
│   ├── train.py                 # Script huấn luyện
│   ├── train_kaggle.py          # Training cho Kaggle
│   ├── dataset.py               # Dataset loader cơ bản
│   ├── dataset_augmented.py     # Dataset với augmentation
│   ├── inference.py             # Inference & LLM integration
│   ├── evaluate_model.py        # Đánh giá mô hình
│   └── devlog.txt               # Lịch sử phát triển
│
├── 📁 STT/                      # Speech-to-Text Module
│   └── stt_engine.py            # STT engine (Whisper/Vosk)
│
├── 📁 LLM/                      # Large Language Model Module
│   └── llm_engine.py            # LLM engine (Groq, Gemini, etc.)
│
├── 📁 TTS/                      # Text-to-Speech Module
│   ├── __init__.py              # Module exports
│   └── tts_engine.py            # TTS engine (ElevenLabs, Edge, gTTS)
│
├── 📁 Pipeline/                 # EVA Pipeline Orchestrator
│   └── eva_pipeline.py          # Main pipeline (STT + SER + LLM + TTS)
│
├── 📁 API/                      # REST API Server
│   └── eva_api.py               # FastAPI backend
│
├── 📁 Dataset/                  # Scripts quản lý dataset
│   ├── download_datasets.py     # Tải RAVDESS, TESS, CREMA-D
│   ├── prepare_dataset.py       # Chuẩn bị & chia dataset
│   ├── extract_dataset.py       # Giải nén manual
│   ├── extract_dataset_colab.py # Cho Google Colab
│   └── kaggle_organize_datasets.py  # Cho Kaggle
│
├── 📁 prompts/                  # Prompt templates
│   ├── system_context.txt       # System context for LLM
│   └── response_guidelines.txt  # Emotion-specific guidelines
│
├── 📁 EVA_Dataset/              # Dataset đã xử lý
│   ├── processed_audio/         # Audio files (.wav)
│   └── labels/                  # Label files (.csv)
│
├── 📁 checkpoints/              # Model checkpoints
├── 📁 logs/                     # Training logs
├── 📁 plots/                    # Visualization plots
├── 📁 evaluation_results/       # Kết quả đánh giá
│
├── 📄 .env.example              # Environment variables template
├── 📄 requirements.txt          # Dependencies
└── 📄 README.md                 # Documentation
```

---

## 🚀 Hướng dẫn Sử dụng

### 1️⃣ Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

**Thư viện chính:**
- `torch>=2.0.0` - Deep learning framework
- `librosa>=0.10.0` - Audio processing
- `openai-whisper` - Speech-to-text
- `groq`, `google-generativeai` - LLM APIs
- `elevenlabs` - Text-to-speech (premium)
- `edge-tts` - Text-to-speech (free)
- `fastapi`, `uvicorn` - REST API
- `numpy`, `pandas`, `scikit-learn` - Data processing
- `tqdm`, `matplotlib`, `seaborn` - Utilities & visualization

### 1.5️⃣ Cấu hình API Keys

```bash
# Copy template và điền API keys
cp .env.example .env

# Edit .env với API keys của bạn
# - GROQ_API_KEY hoặc GEMINI_API_KEY (cho LLM)
# - ELEVENLABS_API_KEY (cho TTS - tùy chọn)
```

### 2️⃣ Chuẩn bị Dataset

#### Option A: Tự động download (khuyến nghị)

```bash
python Dataset/download_datasets.py
```

Dataset sẽ tải về:
- **RAVDESS** (~1.5GB) - 1,440 samples
- **TESS** (~400MB) - 2,800 samples  
- **CREMA-D** (~2GB) - 7,442 samples

#### Option B: Manual download

1. Download các dataset từ nguồn chính thức
2. Đặt file zip vào thư mục `Dataset/`
3. Chạy:
```bash
python Dataset/extract_dataset.py
```

#### Xử lý và chia dataset

```bash
python Dataset/prepare_dataset.py
```

Kết quả:
- ✅ Train set (70%): ~8,177 samples
- ✅ Val set (15%): ~1,752 samples
- ✅ Test set (15%): ~1,753 samples

### 3️⃣ Huấn luyện Mô hình

```bash
python VAE/train.py
```

**Cấu hình mặc định:**
- Batch size: 32
- Learning rate: 1e-3
- Epochs: 100 (với early stopping)
- Latent dim: 64
- Augmentation: ON

**Theo dõi training:**
```bash
# Xem plots
ls plots/

# Xem logs
cat logs/training_history.json
```

### 4️⃣ Đánh giá Mô hình

```bash
python VAE/evaluate_model.py
```

Kết quả sẽ được lưu trong `evaluation_results/`:
- 📊 Per-class metrics
- 📈 Confusion matrices
- 🔍 Latent space visualization
- 📄 Classification report

### 5️⃣ Inference - Sử dụng Mô hình

```python
from VAE.inference import EmotionRecognizer

# Khởi tạo
recognizer = EmotionRecognizer(
    checkpoint_path="checkpoints/best_model.pth",
    device='cuda'
)

# Phân tích audio
emotions, dominant, latent = recognizer.predict("audio_file.wav")

# Tạo LLM prompt
user_text = "Tôi cảm thấy rất mệt mỏi..."
prompt = recognizer.generate_llm_prompt(user_text, emotions)
```

### 6️⃣ Sử dụng Full Pipeline (STT + SER + LLM + TTS)

```bash
# Xử lý audio file với output audio
python Pipeline/eva_pipeline.py audio.wav --output-audio response.mp3

# Sử dụng ElevenLabs TTS
python Pipeline/eva_pipeline.py audio.wav --tts-backend elevenlabs

# Sử dụng Edge TTS (miễn phí)
python Pipeline/eva_pipeline.py audio.wav --tts-backend edge
```

### 7️⃣ Chạy API Server

```bash
# Khởi động server
python API/eva_api.py

# Hoặc với uvicorn
uvicorn API.eva_api:app --host 0.0.0.0 --port 8000 --reload
```

**API Endpoints:**
- `POST /process` - Full pipeline (STT + SER + LLM)
- `POST /process/with-audio` - Full pipeline với TTS audio response
- `POST /synthesize` - Text-to-speech (standalone)
- `GET /tts/voices` - Danh sách voices
- `GET /docs` - Interactive API documentation

---

## 🔊 Module TTS - Text-to-Speech

### Backends hỗ trợ

| Backend | Chất lượng | Vietnamese | API Key | Ghi chú |
|---------|-----------|------------|---------|---------|
| **ElevenLabs** | Tốt nhất | ✅ | Required | Multilingual v2 |
| **Edge TTS** | Tốt | ✅ | Free | Microsoft voices |
| **gTTS** | Cơ bản | ✅ | Free | Google Translate |

### Cấu hình ElevenLabs (Khuyến nghị)

```bash
# Lấy API key tại: https://elevenlabs.io/
export ELEVENLABS_API_KEY=your_key_here

# Hoặc thêm vào .env
TTS_BACKEND=elevenlabs
ELEVENLABS_API_KEY=your_key_here
```

### Sử dụng TTS Engine

```python
from TTS.tts_engine import TTSEngine

# Khởi tạo (auto-detect backend)
tts = TTSEngine(language="vi")

# Synthesize text
response = tts.synthesize("Xin chào! Tôi là EVA.")

# Lưu audio
with open("output.mp3", "wb") as f:
    f.write(response.audio_data)

# Hoặc lưu trực tiếp
tts.synthesize_to_file("Xin chào!", "output.mp3")
```

### Vietnamese Voices

**ElevenLabs:**
- Sử dụng model `eleven_multilingual_v2` (tự động)
- Hỗ trợ nhiều giọng: Adam, Rachel, và nhiều hơn

**Edge TTS:**
- `vi-VN-HoaiMyNeural` (Female)
- `vi-VN-NamMinhNeural` (Male)

---

## 📊 Tiền xử lý Audio

### Mel Spectrogram Configuration

```python
sr = 16000              # Sample rate
n_mels = 128            # Mel frequency bins
hop_length = 512        # Frame shift
n_fft = 2048           # FFT window size
duration = 3            # Audio length (seconds)
```

### Data Augmentation (Training only)

- ⏱️ Time shifting
- 🔊 Noise injection
- 🎵 Speed perturbation
- 📉 Time masking (SpecAugment)
- 📊 Frequency masking (SpecAugment)

---

## 🎓 Training trên Cloud Platforms

### Google Colab

```python
# Upload file colab_extract_datasets.py
!python Dataset/extract_dataset_colab.py

# Prepare dataset
!python Dataset/prepare_dataset.py

# Train
!python VAE/train.py
```

### Kaggle

```python
# Add datasets via UI (RAVDESS, TESS, CREMA-D)
!python Dataset/kaggle_organize_datasets.py

# Prepare & train
!python Dataset/prepare_dataset.py
!python VAE/train_kaggle.py
```

---

## 🔬 Roadmap

### ✅ Giai đoạn 1: SER Model
- [x] Thiết kế kiến trúc Beta-VAE
- [x] Dataset pipeline (RAVDESS, TESS, CREMA-D)
- [x] Training với augmentation
- [x] Evaluation metrics

### ✅ Giai đoạn 2: End-to-End Prototype (Hoàn thành)
- [x] Tích hợp STT (Whisper/Vosk)
- [x] LLM integration (Groq, Gemini, OpenRouter, Ollama)
- [x] Context-aware prompt engine
- [x] TTS module (ElevenLabs, Edge TTS, gTTS)
- [x] REST API (FastAPI)

### 🚧 Giai đoạn 3: Production Ready
- [ ] Web/Mobile UI
- [ ] Dataset tiếng Việt
- [ ] Fine-tuning cho use case cụ thể
- [ ] A/B testing & user feedback

---

## ⚠️ Thách thức & Hạn chế

| Thách thức | Mô tả | Giải pháp đề xuất |
|-----------|-------|-------------------|
| 🗂️ **Dữ liệu** | Thiếu dataset tiếng Việt có gán nhãn cảm xúc | Thu thập & gán nhãn dữ liệu nội bộ |
| ⚖️ **Cân bằng Loss** | Điều chỉnh α, β, γ phức tạp | Grid search + beta warmup |
| ⏱️ **Độ trễ** | Pipeline nhiều bước gây delay | Tối ưu hóa inference, model compression |
| 📏 **Đánh giá** | Thấu cảm khó đo lường định lượng | User studies + qualitative metrics |
| 🎭 **Cảm xúc phức tạp** | Không nắm bắt được cảm xúc xã hội bậc cao | Mở rộng taxonomy, larger models |

---

## 🛠️ Công nghệ Sử dụng

### Core Technologies

| Component | Technology | Version |
|-----------|-----------|---------|
| 🧠 **Deep Learning** | PyTorch | 2.0+ |
| 🎵 **Audio Processing** | Librosa | 0.10+ |
| 🗣️ **STT** | Whisper / Vosk | Latest |
| 🤖 **LLM** | Groq, Gemini, OpenRouter, Ollama | Latest |
| 🔊 **TTS** | ElevenLabs, Edge TTS, gTTS | Latest |
| 📊 **Data Science** | NumPy, Pandas, Scikit-learn | Latest |
| 📈 **Visualization** | Matplotlib, Seaborn | Latest |

### Deployment Stack (Planned)

- 🐳 **Container**: Docker
- 🌐 **API**: FastAPI + Uvicorn
- 📱 **Mobile**: React Native
- ☁️ **Cloud**: AWS/GCP/Azure

---

## 📚 Tài liệu tham khảo

### Datasets
- [RAVDESS](https://zenodo.org/record/1188976) - Ryerson Audio-Visual Database
- [TESS](https://tspace.library.utoronto.ca/handle/1807/24487) - Toronto Emotional Speech Set
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) - Crowd Sourced Emotional Multimodal Actors

### Papers
- Higgins et al. (2017) - "β-VAE: Learning Basic Visual Concepts"
- Mirsamadi et al. (2017) - "Automatic Speech Emotion Recognition"

---

## 👥 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

---

## 📄 License

Project này được phát triển cho mục đích nghiên cứu và giáo dục.

---

## 📧 Liên hệ

Nếu có câu hỏi hoặc góp ý, vui lòng tạo issue trên GitHub repository.

---

<div align="center">

**Made with ❤️ for mental health support**

⭐ Nếu project này hữu ích, hãy cho một star!

</div>
