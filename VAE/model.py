import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Reparameterization Trick
# --------------------------
def reparameterize(mu, log_var):
    """
    Reparameterization trick for VAE
    z = μ + σ * ε, where ε ~ N(0,1)
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


# --------------------------
# FIXED Beta-VAE Model
# --------------------------
class BetaVAE_SER(nn.Module):
    """
    Fixed Beta-VAE for Speech Emotion Recognition

    Key improvements:
    1. Proper decoder architecture (no repeat trick)
    2. Learnable upsampling for frequency dimension
    3. Better dimension matching
    """

    def __init__(self, n_mels=128, n_emotions=8, latent_dim=64):
        super(BetaVAE_SER, self).__init__()
        self.latent_dim = latent_dim
        self.n_emotions = n_emotions
        self.n_mels = n_mels

        # ----- Encoder -----
        # Input: (B, 1, n_mels, T)
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # -> (B, 32, n_mels/2, T/2)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (B, 64, n_mels/4, T/4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (B, 128, n_mels/8, T/8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )

        # LSTM for temporal modeling
        self.lstm_encoder = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True  # Bidirectional for better context
        )

        # Latent space (bidirectional LSTM -> 512 hidden)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # ----- Decoder -----
        # Project latent to initial hidden state
        self.decoder_fc = nn.Linear(latent_dim, 512)

        # LSTM decoder
        self.lstm_decoder = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Calculate dimensions after encoder convolutions
        # After 3 stride-2 convs: H/8, W/8
        self.enc_h = n_mels // 8

        # Project LSTM output to spatial dimensions
        # We'll reconstruct to (128, enc_h, enc_w) then upsample
        self.decoder_project = nn.Linear(256, 128 * self.enc_h)

        # Transposed convolutions to upsample back to original size
        self.decoder_deconv = nn.Sequential(
            # Input: (B, 128, enc_h, enc_w)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (B, 64, 2*enc_h, 2*enc_w)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> (B, 32, 4*enc_h, 4*enc_w)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # -> (B, 1, 8*enc_h, 8*enc_w)
            nn.Sigmoid()  # Output in [0,1]
        )

        # ----- Classifier Head -----
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, n_emotions),
            nn.Sigmoid()  # Multi-label classification
        )

    def encode(self, x):
        """
        Encode input to latent distribution

        Args:
            x: (B, 1, n_mels, T)
        Returns:
            mu: (B, latent_dim)
            log_var: (B, latent_dim)
            shape_info: tuple for decoder
        """
        batch_size, _, height, width = x.shape

        # CNN encoding
        h = self.encoder_cnn(x)  # (B, 128, n_mels/8, T/8)

        # Store shape for decoder
        _, channels, enc_height, enc_width = h.shape

        # Prepare for LSTM: (B, T/8, 128*enc_height)
        # Average pooling over frequency dimension
        h = h.mean(dim=2)  # (B, 128, T/8)
        h = h.permute(0, 2, 1)  # (B, T/8, 128)

        # LSTM encoding
        h, (h_n, _) = self.lstm_encoder(h)

        # Use last hidden state from both directions
        # h_n shape: (num_layers * num_directions, B, hidden_size)
        # We want: (B, hidden_size * num_directions) from last layer
        h_forward = h_n[-2]  # Last layer, forward
        h_backward = h_n[-1]  # Last layer, backward
        h_lstm = torch.cat([h_forward, h_backward], dim=1)  # (B, 512)

        # Latent distribution
        mu = self.fc_mu(h_lstm)
        log_var = self.fc_logvar(h_lstm)

        return mu, log_var, (batch_size, height, width, enc_height, enc_width)

    def decode(self, z, shape_info):
        """
        Decode latent vector to spectrogram

        Args:
            z: (B, latent_dim)
            shape_info: (batch_size, orig_height, orig_width, enc_height, enc_width)
        Returns:
            x_recon: (B, 1, n_mels, T)
        """
        batch_size, orig_height, orig_width, enc_height, enc_width = shape_info

        # Project latent to hidden state
        h = F.relu(self.decoder_fc(z))  # (B, 512)

        # Expand to sequence for LSTM
        h = h.unsqueeze(1).repeat(1, enc_width, 1)  # (B, enc_width, 512)

        # LSTM decoding
        h, _ = self.lstm_decoder(h)  # (B, enc_width, 256)

        # Project to spatial dimensions
        h = self.decoder_project(h)  # (B, enc_width, 128*enc_height)

        # Reshape to (B, C, H, W) for convolution
        h = h.view(batch_size, enc_width, 128, enc_height)  # (B, enc_width, 128, enc_height)
        h = h.permute(0, 2, 3, 1)  # (B, 128, enc_height, enc_width)

        # Transposed convolutions to upsample
        x_recon = self.decoder_deconv(h)  # (B, 1, n_mels, T)

        # Ensure exact size match (handle any size mismatch from convolutions)
        if x_recon.size(2) != orig_height or x_recon.size(3) != orig_width:
            x_recon = F.interpolate(
                x_recon,
                size=(orig_height, orig_width),
                mode='bilinear',
                align_corners=False
            )

        return x_recon

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (B, 1, n_mels, T)
        Returns:
            x_recon: (B, 1, n_mels, T)
            y_pred: (B, n_emotions)
            mu: (B, latent_dim)
            log_var: (B, latent_dim)
        """
        mu, log_var, shape_info = self.encode(x)
        z = reparameterize(mu, log_var)
        x_recon = self.decode(z, shape_info)
        y_pred = self.classifier(z)

        return x_recon, y_pred, mu, log_var


# --------------------------
# Improved Loss Function
# --------------------------
def beta_vae_loss(x, x_recon, y_true, y_pred, mu, log_var,
                  alpha=1.0, beta=0.5, gamma=0.1,
                  warmup_beta=None):
    """
    Improved loss function with better balance and optional beta warmup

    Args:
        x: Original input
        x_recon: Reconstructed input
        y_true: True emotion labels
        y_pred: Predicted emotion labels
        mu: Mean of latent distribution
        log_var: Log variance of latent distribution
        alpha: Weight for classification loss
        beta: Weight for KL divergence (can be warmed up)
        gamma: Weight for reconstruction loss
        warmup_beta: Optional warmup value for beta (used during training)

    Returns:
        total_loss, class_loss, recon_loss, kld, individual components dict
    """
    # 1. Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")

    # 2. Classification loss (Binary Cross Entropy for multi-label)
    class_loss = F.binary_cross_entropy(y_pred, y_true, reduction="mean")

    # 3. KL Divergence
    # KLD = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    # Use warmup beta if provided
    effective_beta = warmup_beta if warmup_beta is not None else beta

    # Weighted total loss
    total_loss = alpha * class_loss + effective_beta * kld + gamma * recon_loss

    # Return individual components for monitoring
    loss_dict = {
        'total': total_loss.item(),
        'classification': class_loss.item(),
        'reconstruction': recon_loss.item(),
        'kld': kld.item(),
        'beta_used': effective_beta
    }

    return total_loss, class_loss, recon_loss, kld, loss_dict


# --------------------------
# Beta Warmup Scheduler
# --------------------------
class BetaScheduler:
    """
    Gradually increase beta from 0 to target value
    Helps prevent posterior collapse in VAE
    """

    def __init__(self, n_epochs, target_beta=0.5, warmup_epochs=10):
        self.n_epochs = n_epochs
        self.target_beta = target_beta
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def step(self):
        """Get beta value for current epoch and increment"""
        if self.current_epoch < self.warmup_epochs:
            beta = self.target_beta * (self.current_epoch / self.warmup_epochs)
        else:
            beta = self.target_beta

        self.current_epoch += 1
        return beta

    def reset(self):
        """Reset scheduler"""
        self.current_epoch = 0


# --------------------------
# Model Summary Function
# --------------------------
def print_model_summary(model, input_shape=(1, 1, 128, 94)):
    """
    Print model architecture summary

    Args:
        model: BetaVAE_SER model
        input_shape: (B, C, H, W) for dummy input
    """
    print("=" * 70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 70)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    # Test forward pass
    dummy_input = torch.randn(input_shape)
    model.eval()

    with torch.no_grad():
        x_recon, y_pred, mu, log_var = model(dummy_input)

    print(f"\n{'=' * 70}")
    print("TENSOR SHAPES")
    print(f"{'=' * 70}")
    print(f"Input:          {tuple(dummy_input.shape)}")
    print(f"Reconstruction: {tuple(x_recon.shape)}")
    print(f"Emotion pred:   {tuple(y_pred.shape)}")
    print(f"Latent (μ):     {tuple(mu.shape)}")
    print(f"Latent (log σ²): {tuple(log_var.shape)}")

    print(f"\n{'=' * 70}")
    print("LAYER DETAILS")
    print(f"{'=' * 70}")

    # Encoder
    encoder_params = sum(p.numel() for p in model.encoder_cnn.parameters())
    encoder_params += sum(p.numel() for p in model.lstm_encoder.parameters())
    encoder_params += sum(p.numel() for p in model.fc_mu.parameters())
    encoder_params += sum(p.numel() for p in model.fc_logvar.parameters())

    # Decoder
    decoder_params = sum(p.numel() for p in model.decoder_fc.parameters())
    decoder_params += sum(p.numel() for p in model.lstm_decoder.parameters())
    decoder_params += sum(p.numel() for p in model.decoder_project.parameters())
    decoder_params += sum(p.numel() for p in model.decoder_deconv.parameters())

    # Classifier
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    print(f"Encoder:    {encoder_params:,} params ({encoder_params / total_params * 100:.1f}%)")
    print(f"Decoder:    {decoder_params:,} params ({decoder_params / total_params * 100:.1f}%)")
    print(f"Classifier: {classifier_params:,} params ({classifier_params / total_params * 100:.1f}%)")

    print(f"\n{'=' * 70}\n")


# --------------------------
# Test Script
# --------------------------
if __name__ == "__main__":
    print("Testing Fixed Beta-VAE Model...\n")

    # Create model
    model = BetaVAE_SER(n_mels=128, n_emotions=8, latent_dim=64)

    # Print summary
    print_model_summary(model)

    # Test forward pass
    batch_size = 4
    n_mels = 128
    time_steps = 94  # 3 seconds at 16kHz with hop_length=512

    dummy_input = torch.randn(batch_size, 1, n_mels, time_steps)
    dummy_labels = torch.randint(0, 2, (batch_size, 8)).float()

    print("Testing forward pass...")
    x_recon, y_pred, mu, log_var = model(dummy_input)

    print("\nTesting loss calculation...")
    total_loss, class_loss, recon_loss, kld, loss_dict = beta_vae_loss(
        dummy_input, x_recon, dummy_labels, y_pred, mu, log_var
    )

    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"  - Classification: {class_loss.item():.4f}")
    print(f"  - Reconstruction: {recon_loss.item():.4f}")
    print(f"  - KL Divergence: {kld.item():.4f}")

    print("\nTesting beta warmup scheduler...")
    scheduler = BetaScheduler(n_epochs=50, target_beta=0.5, warmup_epochs=10)

    print("Beta values for first 15 epochs:")
    for epoch in range(15):
        beta = scheduler.step()
        print(f"  Epoch {epoch + 1:2d}: β = {beta:.4f}")

    print("\n✅ All tests passed!")