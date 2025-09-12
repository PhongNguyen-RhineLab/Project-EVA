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
# Beta-VAE Multi-task Model
# --------------------------
class BetaVAE_SER(nn.Module):
    def __init__(self, n_mels=128, n_emotions=8, latent_dim=32):
        super(BetaVAE_SER, self).__init__()
        self.latent_dim = latent_dim
        self.n_emotions = n_emotions

        # ----- Encoder -----
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, n_mels/2, T/2)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (B, 64, n_mels/4, T/4)
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
        self.decoder_deconv = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

        # ----- Classifier Head -----
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_emotions),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_cnn(x)  # (B, C, H, W)
        h = h.mean(dim=[2])      # Global avg pooling over frequency
        h = h.permute(0,2,1)     # (B, T, C)
        _, (h_lstm, _) = self.lstm(h)
        h_lstm = h_lstm[-1]      # (B, 128)

        mu = self.fc_mu(h_lstm)
        log_var = self.fc_logvar(h_lstm)
        return mu, log_var

    def decode(self, z):
        h = F.relu(self.decoder_fc(z))  # (B, 128)
        h = h.unsqueeze(1).repeat(1, 10, 1)  # giả định chuỗi dài 10
        h, _ = self.decoder_lstm(h)
        h = h.unsqueeze(-1).permute(0,2,1,3)  # (B,64,T,1)
        x_recon = torch.sigmoid(self.decoder_deconv(h))  # (B,1,H,W)
        return x_recon

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = reparameterize(mu, log_var)
        x_recon = self.decode(z)
        y_pred = self.classifier(z)
        return x_recon, y_pred, mu, log_var

# --------------------------
# Loss Function
# --------------------------
def beta_vae_loss(x, x_recon, y_true, y_pred, mu, log_var, alpha=1.0, beta=1.0, gamma=1.0):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")

    # Classification loss (Binary Crossentropy)
    class_loss = F.binary_cross_entropy(y_pred, y_true, reduction="mean")

    # KL Divergence
    kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    # Weighted total loss
    total_loss = alpha * class_loss + beta * kld + gamma * recon_loss
    return total_loss, class_loss, recon_loss, kld
