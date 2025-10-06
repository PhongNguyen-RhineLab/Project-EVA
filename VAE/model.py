import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Reparameterization Trick
# --------------------------
def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


# --------------------------
# Beta-VAE Multi-task Model (FIXED)
# --------------------------
class BetaVAE_SER(nn.Module):
    def __init__(self, n_mels=128, n_emotions=8, latent_dim=32):
        super(BetaVAE_SER, self).__init__()
        self.latent_dim = latent_dim
        self.n_emotions = n_emotions
        self.n_mels = n_mels

        # ----- Encoder -----
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, n_mels/2, T/2)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, n_mels/4, T/4)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Flatten + LSTM
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)

        # Latent space
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # ----- Decoder -----
        self.decoder_fc = nn.Linear(latent_dim, 128)
        self.decoder_lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)

        # Calculate spatial dimensions after convolutions
        # After 2 stride-2 convs: H/4, W/4
        self.decoder_h = n_mels // 4

        # Deconvolution layers to match encoder (reversed)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0,1] to match normalized input
        )

        # ----- Classifier Head -----
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_emotions),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Input: (B, 1, n_mels, T)
        Output: mu, log_var (both B x latent_dim), and shape info
        """
        batch_size, _, height, width = x.shape

        h = self.encoder_cnn(x)  # (B, 64, n_mels/4, T/4)

        # Store original shape for decoder
        _, channels, enc_height, enc_width = h.shape

        # Global avg pooling over frequency dimension
        h = h.mean(dim=2)  # (B, 64, T/4)
        h = h.permute(0, 2, 1)  # (B, T/4, 64)

        _, (h_lstm, _) = self.lstm(h)
        h_lstm = h_lstm[-1]  # (B, 128)

        mu = self.fc_mu(h_lstm)
        log_var = self.fc_logvar(h_lstm)

        return mu, log_var, (batch_size, height, width, enc_height, enc_width)

    def decode(self, z, shape_info):
        """
        Input:
            z: (B, latent_dim)
            shape_info: (batch_size, orig_height, orig_width, enc_height, enc_width)
        Output: (B, 1, n_mels, T) - matching input shape
        """
        batch_size, orig_height, orig_width, enc_height, enc_width = shape_info

        h = F.relu(self.decoder_fc(z))  # (B, 128)

        # Expand to sequence matching encoder's temporal dimension
        h = h.unsqueeze(1).repeat(1, enc_width, 1)  # (B, enc_width, 128)

        h, _ = self.decoder_lstm(h)  # (B, enc_width, 64)

        # Reshape for deconvolution: (B, C, H, W)
        h = h.permute(0, 2, 1)  # (B, 64, enc_width)
        h = h.unsqueeze(2).repeat(1, 1, enc_height, 1)  # (B, 64, enc_height, enc_width)

        # Deconvolve back to original dimensions
        x_recon = self.decoder_deconv(h)  # (B, 1, n_mels, T)

        # Ensure exact match with input size (crop or pad if needed)
        if x_recon.size(2) != orig_height or x_recon.size(3) != orig_width:
            x_recon = F.interpolate(x_recon, size=(orig_height, orig_width), mode='bilinear', align_corners=False)

        return x_recon

    def forward(self, x):
        mu, log_var, shape_info = self.encode(x)
        z = reparameterize(mu, log_var)
        x_recon = self.decode(z, shape_info)
        y_pred = self.classifier(z)
        return x_recon, y_pred, mu, log_var


# --------------------------
# Loss Function
# --------------------------
def beta_vae_loss(x, x_recon, y_true, y_pred, mu, log_var, alpha=1.0, beta=0.5, gamma=0.1):
    """
    UPDATED: Better default weights for loss balance
    - alpha=1.0: Classification (BCE, typically 0.1-0.7)
    - beta=0.5: KL divergence (typically 0.001-0.1)
    - gamma=0.1: Reconstruction (MSE, typically 0.01-0.5)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")

    # Classification loss (Binary Crossentropy)
    class_loss = F.binary_cross_entropy(y_pred, y_true, reduction="mean")

    # KL Divergence
    kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    # Weighted total loss
    total_loss = alpha * class_loss + beta * kld + gamma * recon_loss
    return total_loss, class_loss, recon_loss, kld