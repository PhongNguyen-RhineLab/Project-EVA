# Project-EVA
Empathic Voice Assistant (EVA) to help people with psychological disorder  

Trợ lý ảo Thấu cảm sử dụng Mô hình Ngôn ngữ Lớn và Phân tích Giọng nói Đa nhãn
1. Mục tiêu chính
Xây dựng một hệ thống AI hoàn chỉnh có khả năng:
1.	Hiểu nội dung văn bản của người nói qua việc chuyển đổi giọng nói thành văn bản (Speech-to-Text).
2.	Cảm nhận trạng thái cảm xúc phức hợp của người nói bằng cách phân tích các đặc tính âm học của giọng nói, cho ra một vector xác suất đa nhãn (ví dụ: Vui 30%, Buồn 70%).
3.	Phản hồi một cách thông minh và thấu cảm bằng cách sử dụng một mô hình ngôn ngữ lớn (LLM) kết hợp với thông tin về cả nội dung và cảm xúc.
2. Kiến trúc Hệ thống Tổng thể
Luồng xử lý dữ liệu sẽ đi theo chu trình sau:
Giọng nói người dùng -> [Xử lý song song]
1.	Module Speech-to-Text (STT) -> Văn bản nội dung
2.	Module Nhận diện Cảm xúc (Speak Emotion Recognition – SER) -> Vector cảm xúc
[Tổng hợp] -> Bộ tạo Prompt Thích ứng -> Prompt hoàn chỉnh -> LLM -> Văn bản phản hồi -> Module Text-to-Speech (TTS) -> Âm thanh phản hồi
3. Chi tiết Kỹ thuật các Module chính
Module 1: Thu thập và Tiền xử lý Dữ liệu
•	Nguồn dữ liệu:
o	Bắt đầu: Sử dụng các bộ dữ liệu tiếng Anh công khai có chất lượng cao như RAVDESS, IEMOCAP, Emo-DB.
o	Mục tiêu dài hạn: Xây dựng hoặc tìm kiếm bộ dữ liệu tiếng Việt được gán nhãn cảm xúc để mô hình có tính ứng dụng cao tại Việt Nam. (tiếng việt để + điểm)
•	Tiền xử lý:
o	Chuyển đổi: Mỗi file audio sẽ được chuyển đổi thành Mel Spectrogram.
o	Tham số đề xuất: n_mels=128, hop_length=512, n_fft=2048. Cần chuẩn hóa (normalize) giá trị của spectrogram về khoảng [0, 1] hoặc [-1, 1].
•	Gán nhãn (Labeling):
o	Quan trọng: Vì mục tiêu là phân loại đa nhãn, nhãn cho mỗi file audio phải ở dạng vector. Ví dụ, nếu có 4 cảm xúc [Vui, Buồn, Tức giận, Trung tính], một file audio Buồn sẽ có nhãn là [0, 1, 0, 0]. Một file vừa Vui vừa Ngạc nhiên có thể là [1, 0, 0, 0, 1].
o	Bộ 8 cảm xúc chính:
	Trung tính (Neutral)
	Bình tĩnh (Calm)
	Vui vẻ (Happy)
	Buồn bã (Sad)
	Tức giận (Angry)
	Sợ hãi (Fearful)
	Ghê tởm (Disgust)
	Ngạc nhiên (Surprised)
*Mạnh theo lý thuyết nhưng vẫn có những giới hạn. Nó khó có thể nắm bắt được các cảm xúc xã hội hoặc nhận thức bậc cao, những cảm xúc phụ thuộc nhiều vào bối cảnh văn hóa và suy nghĩ nội tâm hơn là đặc tính âm học của giọng nói. Ví dụ:
•	Tội lỗi (Guilt)
•	Xấu hổ (Shame)
•	Tự hào (Pride)
•	Ghen tị (Jealousy)
	Hướng cải thiện nếu bị hỏi 

Module 2: Mô hình Học Đặc trưng và Phân loại Cảm xúc (SER)
•	Tên mô hình: Beta-VAE Đa nhiệm cho Cảm xúc (Multi-task Beta-VAE for Emotions).
•	Kiến trúc lõi:
1.	Encoder:
	Kiến trúc đề xuất: Hybrid CNN + LSTM.
	Cấu trúc ví dụ:
	3-4 lớp Conv2D (với ReLU, Batch Normalization) để học các đặc trưng cục bộ từ spectrogram.
	MaxPooling2D sau mỗi lớp Conv2D.
	Flatten/Reshape để chuyển output 2D thành chuỗi 1D.
	1-2 lớp LSTM hoặc GRU để học sự phụ thuộc theo thời gian của chuỗi.
2.	Latent Space:
	Output của Encoder được đưa vào 2 lớp Dense để tạo ra mu (vector trung bình) và log_var (vector log phương sai).
	Lấy mẫu vector tiềm ẩn z từ mu và log_var.
	Kích thước z đề xuất: Bắt đầu với 32, có thể tăng lên 64 hoặc 128.
3.	Decoder (Nhánh Tái tạo):
	Lấy z làm đầu vào, sử dụng kiến trúc ngược với Encoder (ví dụ: LSTM -> Upsampling -> Transposed Conv2D) để tái tạo lại Mel Spectrogram.
4.	Classifier Head (Nhánh Phân loại):
	Lấy output của Encoder (trước bước tạo mu, log_var) hoặc chính z làm đầu vào.
	1-2 lớp Dense với Dropout để tránh overfitting.
	Lớp Output: Lớp Dense với n neuron (n là số cảm xúc) và hàm kích hoạt sigmoid.
•	Hàm Mất mát (Loss Function):
*Note: Đây là công thức tóm tắt, cần tinh chỉnh các trọng số α, β, γ:
L_total = alpha * L_classification + beta * L_kld + gamma * L_reconstruction
L_classification = BinaryCrossentropy(y_true, y_pred_classifier)
L_reconstruction = MeanSquaredError(x_true, x_reconstructed)
L_kld = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
Module 3: Tích hợp LLM (Gemma)
•	Tên module: Bộ tạo Prompt Thích ứng Ngữ cảnh (Context-Adaptive Prompt Engine).
•	Logic: Xây dựng một hệ thống (có thể là các template đơn giản ban đầu) để ghép nối thông tin.
•	Template Prompt:
•	"Bối cảnh hệ thống: [Ví dụ: Bạn là một trợ lý ảo thấu cảm.]
•	Phân tích cảm xúc từ giọng nói: [Mô tả vector cảm xúc một cách tự nhiên. Ví dụ: Người dùng có vẻ đang cảm thấy buồn bã (70%) và một chút mệt mỏi (40%).]
•	Nhiệm vụ: [Chỉ dẫn cho LLM. Ví dụ: Hãy phản hồi câu nói sau một cách an ủi, tránh đưa ra lời khuyên trực tiếp trừ khi được hỏi.]
•	Nội dung câu nói của người dùng: [Văn bản từ STT.]"
*chưa triển khai
4. Lộ trình Triển khai (Roadmap)
•	Giai đoạn 1: Xây dựng và Huấn luyện Mô hình SER
1.	Chọn và xử lý bộ dữ liệu (ví dụ: RAVDESS).
2.	Implement kiến trúc Beta-VAE Đa nhiệm trên TensorFlow/PyTorch.
3.	Huấn luyện và tinh chỉnh mô hình để đạt được kết quả tốt trên tập validation.
4.	Mục tiêu: Mô hình có khả năng xuất ra vector xác suất cảm xúc có ý nghĩa.
•	Giai đoạn 2: Xây dựng Prototype End-to-End 
1.	Tích hợp các API/mô hình STT và TTS đơn giản.
2.	Xây dựng "Bộ tạo Prompt Thích ứng" phiên bản đầu tiên.
3.	Kết nối với Gemma API.
4.	Mục tiêu: Có một sản phẩm demo chạy được từ đầu đến cuối.
•	Giai đoạn 3: Tinh chỉnh và Đánh giá (...)
1.	Thử nghiệm với nhiều tình huống thực tế.
2.	Tinh chỉnh các template của prompt để câu trả lời của Gemma tự nhiên hơn.
3.	Đánh giá hệ thống không chỉ bằng các chỉ số kỹ thuật mà còn bằng cảm nhận chủ quan của người dùng.
5. Thách thức chính
•	Dữ liệu: Việc tìm kiếm hoặc tự tạo dữ liệu tiếng Việt có gán nhãn đa cảm xúc sẽ rất tốn thời gian.
•	Cân bằng Loss: Việc tinh chỉnh các trọng số α, β, γ sẽ là một quá trình thực nghiệm phức tạp.
•	Độ trễ: Hệ thống có nhiều bước có thể gây ra độ trễ trong giao tiếp. Cần tối ưu hóa hiệu năng sau khi có prototype.
•	Đánh giá: Đo lường sự "thấu cảm" là một bài toán khó, không có chỉ số tuyệt đối.

Công nghệ và kiến thức sử dụng trong dự án:
1. Module Speech-to-Text (STT)
•	Công nghệ:
o	Whisper (OpenAI) – mạnh cho đa ngôn ngữ + tiếng Việt. (top pick)
o	Vosk (offline STT + tiếng Việt).
o	Google Cloud Speech-to-Text hoặc Azure Speech Service (mất phí)
•	Ngôn ngữ lập trình: Python (PyTorch).

2. Module Nhận diện Cảm xúc từ Giọng nói (SER)
•	Tiền xử lý dữ liệu âm thanh:
o	Librosa để trích xuất Mel Spectrogram.
o	NumPy, SciPy để xử lý tín hiệu số.
* Cả hai đều là thư viện python
•	Mô hình học đặc trưng & phân loại:
o	PyTorch để xây dựng Beta-VAE đa nhiệm.
o	Kết hợp CNN + LSTM/GRU cho encoder.
o	Sigmoid output layer cho phân loại đa nhãn.
•	Loss function:
o	Binary Crossentropy (phân loại đa nhãn).
o	KL Divergence (VAE).
o	MSE (tái tạo spectrogram).
•	Khung huấn luyện:
o	PyTorch Lightning hoặc HuggingFace Accelerate để dễ quản lý training/validation. (mất phí / chưa tìm hiểu)
3. Module LLM (Bộ não phản hồi thấu cảm)
•	Mô hình ngôn ngữ lớn (LLM):
o	Gemma (Google) 
o	LLaMA, Mistral, GPT-4, DeepSeek R1 (nếu gemma ko cho kết quả tốt)
•	Công nghệ Prompt Engineering:
o	Xây dựng Context-Adaptive Prompt Engine bằng Python.
o	Có thể dùng LangChain hoặc LlamaIndex để quản lý template prompt. (nâng cao)
4. Module Text-to-Speech (TTS)
* chưa quyết định
5. Kiến trúc Hệ thống & Triển khai
•	Ngôn ngữ chính: Python 
•	Triển khai mô hình:
o	Docker để đóng gói từng module (STT, SER, LLM, TTS).
o	REST API để giao tiếp giữa các module.
•	Frontend / Giao diện người dùng:
o	React Native 



#Triển khai lấy dữ liệu
Giai đoạn 1: Thu thập và Tải Dữ liệu thô
1.	Lựa chọn Nguồn Video:
o	Nên chọn:
	Podcast, Talkshow, Kể chuyện: Thường có chất lượng audio tốt, giọng nói rõ ràng, ít nhạc nền, và biểu cảm tự nhiên.
	Kênh review sản phẩm, vlog dạng độc thoại: Giọng nói của một người, tập trung vào việc truyền đạt.
	Sách nói (Audiobooks): Nguồn audio cực kỳ sạch nhưng cảm xúc thường ở mức độ "diễn", ít tự nhiên.
o	Nên tránh:
	Video ca nhạc, clip phim: Chứa nhiều nhạc nền và hiệu ứng âm thanh.
	Vlog du lịch, tổng hợp tin tức: Thường có nhạc nền rất to, nhiều tiếng ồn môi trường.
o	Tiêu chí: Audio sạch, ít nhạc/nhiễu, giọng nói của người là trọng tâm.
2.	Tải Audio từ YouTube:
o	Công cụ đề xuất: Sử dụng yt-dlp, một công cụ dòng lệnh mạnh mẽ và linh hoạt hơn các trang web tải online.
o	Lệnh ví dụ (tải playlist và chuyển thành file .wav):
Bash
# Cài đặt: pip install yt-dlp ffmpeg
# Lệnh tải:
yt-dlp -f 'ba' -x --audio-format wav "URL "
	-f 'ba': Chọn chất lượng audio tốt nhất (best audio).
	-x --audio-format wav: Trích xuất và chuyển đổi thành định dạng .wav.
Giai đoạn 2: Tiền xử lý và Làm sạch Âm thanh
Đây là giai đoạn quyết định chất lượng của bộ dữ liệu.
3.	Chuẩn hóa Âm thanh (Standardization):
o	Mục tiêu: Đưa tất cả các file audio về cùng một định dạng chuẩn.
o	Tần số lấy mẫu (Sample Rate): Chuyển tất cả về 16000 Hz (hoặc 22050 Hz). 16kHz là tiêu chuẩn cho nhận dạng giọng nói.
o	Số kênh (Channels): Chuyển thành mono (1 kênh).
o	Độ sâu bit (Bit Depth): Chuẩn hóa thành 16-bit PCM.
o	Công cụ: ffmpeg hoặc thư viện librosa trong Python có thể làm việc này.
4.	Lọc nhiễu và Tách giọng (Noise Reduction & Voice Isolation):
o	Audio từ YouTube thường không hoàn hảo. Bước này giúp "làm sạch" giọng nói.
o	Tách nhạc nền: Nếu có nhạc nền nhẹ, bạn có thể thử các mô hình như Spleeter hoặc Demucs.
o	Giảm tiếng ồn: Sử dụng các mô hình chuyên dụng để giảm nhiễu (ví dụ: noisereduce trong Python hoặc các mô hình AI tiên tiến hơn như RNNoise). Đây là bước nâng cao nhưng rất đáng giá.
5.	Phân đoạn Thông minh (Intelligent Segmentation):
o	Như đã thảo luận, bạn cần chia các file dài thành các đoạn ngắn.
o	Sử dụng phương pháp "Chia theo Khoảng lặng" đã đề cập trước đó.
o	Lọc để giữ lại các đoạn có độ dài trong khoảng 2-8 giây.
o	Kết quả của bước này là một thư mục chứa hàng ngàn file audio ngắn, sạch, và đã được chuẩn hóa.
Giai đoạn 3: Dán nhãn (The Bottleneck)
Đây là giai đoạn tốn nhiều công sức nhất. Bạn có hàng ngàn file audio ngắn, giờ phải gán cảm xúc cho chúng.
6.	Xác định Chiến lược:
o	Lý tưởng nhất (nhưng khó): Nghe và dán nhãn thủ công cho tất cả.
o	Thực tế nhất (chiến lược bán giám sát):
1.	Dán nhãn cho một phần nhỏ: Đặt mục tiêu dán nhãn cho khoảng 1,000 - 2,000 file một cách cẩn thận nhất có thể. Đây sẽ là tập dữ liệu có nhãn của bạn.
2.	Sử dụng phần còn lại: Hàng ngàn file còn lại sẽ được dùng làm dữ liệu không nhãn để tiền huấn luyện phần VAE của mô hình.
7.	Sử dụng Công cụ Dán nhãn:
o	Đừng làm thủ công bằng cách đổi tên file. Hãy dùng một công cụ chuyên dụng.
o	Đề xuất:
	Audacity: Miễn phí, có tính năng "Label Track" cho phép bạn nghe và gõ nhãn trực tiếp trên dòng thời gian.
	Label Studio: Lựa chọn mạnh mẽ nhất. Đây là công cụ mã nguồn mở, cho phép bạn tạo giao diện dán nhãn audio rất chuyên nghiệp. Bạn có thể thiết lập 8 nút bấm tương ứng với 8 cảm xúc của EVA.
8.	Lưu kết quả dán nhãn:
o	Kết quả của quá trình này nên là một file .csv hoặc .json.
o	Ví dụ file labels.csv:
Đoạn mã
filename,happy,sad,angry,neutral,calm,fearful,disgust,surprised
chunk_001.wav,0,1,0,0,0,0,0,0
chunk_002.wav,1,0,0,0,0,0,0,1
chunk_003.wav,0,0,0,1,0,0,0,0
...
Giai đoạn 4: Tổ chức và Chia Dữ liệu
9.	Cấu trúc Thư mục:
o	Hãy tổ chức dữ liệu một cách khoa học để dễ quản lý.
o	/EVA_Dataset
o	    /raw_downloads/       # Các file wav dài tải từ YouTube
o	    /processed_audio/     # Thư mục chứa các file chunk 2-8s đã sạch
o	    /labels/
o	        labels.csv        # File chứa nhãn cho các file trong processed_audio
10.	Chia Tập Huấn luyện / Kiểm thử (Train/Test Split):
o	Quan Trọng: Phải đảm bảo tách biệt người nói (speaker-independent).
o	Cách làm:
1.	Nhóm các file theo người nói nếu có thể.
2.	Chia người nói thành các nhóm, ví dụ: 80% người nói cho tập train, 10% cho tập validation, 10% cho tập test.
3.	Tuyệt đối không để các file của cùng một người nói xuất hiện trong cả tập train và tập test. Nếu không, mô hình của bạn sẽ chỉ học cách "nhận diện giọng của người đó" thay vì "nhận diện cảm xúc".

