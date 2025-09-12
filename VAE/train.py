from model import BetaVAE_SER, beta_vae_loss
import torch
import torch.optim as optim
from dataset import EmotionDataset
from torch.utils.data import DataLoader

# --------------------------
# Dataset
# --------------------------
# Khai báo dataset
train_dataset = EmotionDataset(
    audio_dir="EVA_Dataset/processed_audio",
    label_file="EVA_Dataset/labels/train_labels.csv"
)
val_dataset = EmotionDataset(
    audio_dir="EVA_Dataset/processed_audio",
    label_file="EVA_Dataset/labels/val_labels.csv"
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16)

# --------------------------
# Model + Optimizer
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BetaVAE_SER(n_mels=128, n_emotions=8, latent_dim=32).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --------------------------
# Training function
# --------------------------
def train_epoch(model, loader, optimizer, device, alpha=1.0, beta=1.0, gamma=1.0):
    model.train()
    total_loss, total_cls, total_rec, total_kld = 0,0,0,0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        x_recon, y_pred, mu, log_var = model(x)
        loss, cls, rec, kld = beta_vae_loss(x, x_recon, y, y_pred, mu, log_var,
                                            alpha=alpha, beta=beta, gamma=gamma)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls  += cls.item()
        total_rec  += rec.item()
        total_kld  += kld.item()

    n = len(loader)
    return total_loss/n, total_cls/n, total_rec/n, total_kld/n

# --------------------------
# Validation function
# --------------------------
def eval_epoch(model, loader, device, alpha=1.0, beta=1.0, gamma=1.0):
    model.eval()
    total_loss, total_cls, total_rec, total_kld = 0,0,0,0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_recon, y_pred, mu, log_var = model(x)
            loss, cls, rec, kld = beta_vae_loss(x, x_recon, y, y_pred, mu, log_var,
                                                alpha=alpha, beta=beta, gamma=gamma)
            total_loss += loss.item()
            total_cls  += cls.item()
            total_rec  += rec.item()
            total_kld  += kld.item()
    n = len(loader)
    return total_loss/n, total_cls/n, total_rec/n, total_kld/n

# --------------------------
# Run training loop
# --------------------------
n_epochs = 10
for epoch in range(1, n_epochs+1):
    train_loss, train_cls, train_rec, train_kld = train_epoch(model, train_loader, optimizer, device)
    val_loss, val_cls, val_rec, val_kld = eval_epoch(model, val_loader, device)

    print(f"Epoch {epoch:02d} | "
          f"Train Loss: {train_loss:.4f} (Cls {train_cls:.4f}, Rec {train_rec:.4f}, KLD {train_kld:.4f}) | "
          f"Val Loss: {val_loss:.4f} (Cls {val_cls:.4f}, Rec {val_rec:.4f}, KLD {val_kld:.4f})")
