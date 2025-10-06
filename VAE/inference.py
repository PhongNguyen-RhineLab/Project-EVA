import torch
import librosa
import numpy as np
from model_fixed import BetaVAE_SER


class EmotionRecognizer:
    """
    Emotion recognition inference class for EVA project
    """

    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']

        # Initialize model
        self.model = BetaVAE_SER(
            n_mels=128,
            n_emotions=8,
            latent_dim=self.config['latent_dim']
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Emotion labels
        self.emotion_labels = [
            "Neutral", "Calm", "Happy", "Sad",
            "Angry", "Fearful", "Disgust", "Surprised"
        ]

        print(f"Model loaded from {checkpoint_path}")
        print(f"Device: {self.device}")

    def preprocess_audio(self, audio_path, sr=16000, n_mels=128, duration=3,
                         hop_length=512, n_fft=2048):
        """
        Preprocess audio file to mel spectrogram
        """
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr, mono=True)

        # Pad/truncate
        max_len = duration * sr
        if len(y) > max_len:
            y = y[:max_len]
        else:
            y = np.pad(y, (0, max_len - len(y)), mode="constant")

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

        # To tensor
        mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        return mel_tensor

    def predict(self, audio_path, threshold=0.5):
        """
        Predict emotions from audio file

        Returns:
            emotions_dict: Dictionary with emotion probabilities
            latent_vector: Latent representation (for further analysis)
        """
        # Preprocess
        mel_tensor = self.preprocess_audio(audio_path).to(self.device)

        # Inference
        with torch.no_grad():
            _, y_pred, mu, _ = self.model(mel_tensor)

        # Convert to numpy
        probs = y_pred.cpu().numpy()[0]
        latent = mu.cpu().numpy()[0]

        # Create emotion dictionary
        emotions_dict = {
            label: float(prob)
            for label, prob in zip(self.emotion_labels, probs)
        }

        # Get dominant emotions (above threshold)
        dominant_emotions = {
            label: prob
            for label, prob in emotions_dict.items()
            if prob > threshold
        }

        return emotions_dict, dominant_emotions, latent

    def format_emotion_output(self, emotions_dict, dominant_emotions):
        """
        Format emotion output for LLM prompt
        """
        # Sort by probability
        sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)

        # Top 3 emotions
        top_3 = sorted_emotions[:3]

        output = "Emotion Analysis:\n"
        output += "  Top emotions:\n"
        for emotion, prob in top_3:
            output += f"    - {emotion}: {prob * 100:.1f}%\n"

        if dominant_emotions:
            output += "\n  Dominant emotions (>50%):\n"
            for emotion, prob in dominant_emotions.items():
                output += f"    - {emotion}: {prob * 100:.1f}%\n"

        return output

    def generate_llm_prompt(self, user_text, emotions_dict):
        """
        Generate context-aware prompt for LLM (Gemma)
        """
        # Get top 2 emotions
        sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)
        top_emotions = sorted_emotions[:2]

        emotion_desc = ", ".join([
            f"{emotion} ({prob * 100:.0f}%)"
            for emotion, prob in top_emotions
        ])

        prompt = f"""System Context: You are EVA, an empathic voice assistant designed to help people with psychological challenges.

Emotional Analysis from Voice:
The user's voice shows: {emotion_desc}

Guidelines:
- Respond with empathy and understanding
- Acknowledge their emotional state naturally (don't list percentages)
- Provide supportive, non-judgmental responses
- Avoid giving direct advice unless asked
- Use a warm, conversational tone
- Keep responses concise but meaningful

User's message: {user_text}

Your empathic response:"""

        return prompt


# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    # Initialize recognizer
    recognizer = EmotionRecognizer(
        checkpoint_path="checkpoints/best_model.pth",
        device='cuda'
    )

    # Example: Analyze audio file
    audio_file = "example_audio.wav"

    print(f"\nAnalyzing: {audio_file}")
    print("=" * 60)

    # Get predictions
    emotions_dict, dominant_emotions, latent = recognizer.predict(
        audio_file,
        threshold=0.5
    )

    # Display results
    print("\nAll Emotion Probabilities:")
    for emotion, prob in emotions_dict.items():
        bar = "█" * int(prob * 20)
        print(f"  {emotion:12s} [{bar:20s}] {prob * 100:5.1f}%")

    print("\n" + recognizer.format_emotion_output(emotions_dict, dominant_emotions))

    # Generate LLM prompt
    user_text = "I've been feeling really overwhelmed lately"
    llm_prompt = recognizer.generate_llm_prompt(user_text, emotions_dict)

    print("\n" + "=" * 60)
    print("LLM Prompt:")
    print("=" * 60)
    print(llm_prompt)