from model_fixed import BetaVAE_SER, beta_vae_loss
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import EmotionDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
import os
from tqdm import tqdm

# --------------------------
# Configuration
# --------------------------
CONFIG = {
    'batch_size': 16,
    'learning_rate': 1e-3,
    'n_epochs': 50,
    'latent_dim': 64,  # Try 32, 64, or 128
    'alpha': 1.0,  # Classification weight
    'beta': 0.5,  # KL divergence weight (start small, increase gradually)
    'gamma': 0.1,  # Reconstruction weight
    'patience': 10,  # Early stopping patience
    'grad_clip': 1.0,  # Gradient clipping
    'save_dir': 'checkpoints'
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

# --------------------------
# Dataset
# --------------------------
train_dataset = EmotionDataset(
    audio_dir="EVA_Dataset/processed_audio",
    label_file="EVA_Dataset/labels/train_labels.csv"
)
val_dataset = EmotionDataset(
    audio_dir="EVA_Dataset/processed_audio",
    label_file="EVA_Dataset/labels/val_labels.csv"
)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                        num_workers=2, pin_memory=True)

# --------------------------
# Model + Optimizer + Scheduler
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = BetaVAE_SER(n_mels=128, n_emotions=8, latent_dim=CONFIG['latent_dim']).to(device)
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


# --------------------------
# Metrics Calculation
# --------------------------
def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Multi-label classification metrics
    """
    y_pred_binary = (y_pred > threshold).astype(int)
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred_binary

    # Subset accuracy (exact match)
    subset_acc = accuracy_score(y_true_np, y_pred_np)

    # Hamming loss (percentage of wrong labels)
    hamming = hamming_loss(y_true_np, y_pred_np)

    # F1 scores
    f1_micro = f1_score(y_true_np, y_pred_np, average='micro', zero_division=0)
    f1_macro = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    f1_samples = f1_score(y_true_np, y_pred_np, average='samples', zero_division=0)

    return {
        'subset_accuracy': subset_acc,
        'hamming_loss': hamming,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_samples': f1_samples
    }


# --------------------------
# Training function
# --------------------------
def train_epoch(model, loader, optimizer, device, config):
    model.train()
    total_loss, total_cls, total_rec, total_kld = 0, 0, 0, 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc='Training')
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        x_recon, y_pred, mu, log_var = model(x)

        loss, cls, rec, kld = beta_vae_loss(
            x, x_recon, y, y_pred, mu, log_var,
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma']
        )

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        optimizer.step()

        total_loss += loss.item()
        total_cls += cls.item()
        total_rec += rec.item()
        total_kld += kld.item()

        # Collect predictions for metrics
        all_preds.append(y_pred.detach().cpu().numpy())
        all_labels.append(y.cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    n = len(loader)

    # Calculate metrics
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = calculate_metrics(torch.tensor(all_labels), all_preds)

    return {
        'loss': total_loss / n,
        'cls_loss': total_cls / n,
        'rec_loss': total_rec / n,
        'kld_loss': total_kld / n,
        **metrics
    }


# --------------------------
# Validation function
# --------------------------
def eval_epoch(model, loader, device, config):
    model.eval()
    total_loss, total_cls, total_rec, total_kld = 0, 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc='Validation'):
            x, y = x.to(device), y.to(device)
            x_recon, y_pred, mu, log_var = model(x)

            loss, cls, rec, kld = beta_vae_loss(
                x, x_recon, y, y_pred, mu, log_var,
                alpha=config['alpha'],
                beta=config['beta'],
                gamma=config['gamma']
            )

            total_loss += loss.item()
            total_cls += cls.item()
            total_rec += rec.item()
            total_kld += kld.item()

            all_preds.append(y_pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    n = len(loader)

    # Calculate metrics
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = calculate_metrics(torch.tensor(all_labels), all_preds)

    return {
        'loss': total_loss / n,
        'cls_loss': total_cls / n,
        'rec_loss': total_rec / n,
        'kld_loss': total_kld / n,
        **metrics
    }


# --------------------------
# Training Loop with Early Stopping
# --------------------------
best_val_loss = float('inf')
patience_counter = 0

print(f"\n{'=' * 60}")
print(f"Training Configuration:")
print(f"  Latent Dim: {CONFIG['latent_dim']}")
print(f"  Loss Weights - α: {CONFIG['alpha']}, β: {CONFIG['beta']}, γ: {CONFIG['gamma']}")
print(f"  Learning Rate: {CONFIG['learning_rate']}")
print(f"  Batch Size: {CONFIG['batch_size']}")
print(f"{'=' * 60}\n")

for epoch in range(1, CONFIG['n_epochs'] + 1):
    print(f"\n--- Epoch {epoch}/{CONFIG['n_epochs']} ---")

    # Train
    train_metrics = train_epoch(model, train_loader, optimizer, device, CONFIG)

    # Validate
    val_metrics = eval_epoch(model, val_loader, device, CONFIG)

    # Learning rate scheduling
    scheduler.step(val_metrics['loss'])

    # Print metrics
    print(f"\nTrain - Loss: {train_metrics['loss']:.4f} | "
          f"Cls: {train_metrics['cls_loss']:.4f} | "
          f"Rec: {train_metrics['rec_loss']:.4f} | "
          f"KLD: {train_metrics['kld_loss']:.4f}")
    print(f"        F1-Micro: {train_metrics['f1_micro']:.4f} | "
          f"F1-Macro: {train_metrics['f1_macro']:.4f} | "
          f"Subset Acc: {train_metrics['subset_accuracy']:.4f}")

    print(f"\nVal   - Loss: {val_metrics['loss']:.4f} | "
          f"Cls: {val_metrics['cls_loss']:.4f} | "
          f"Rec: {val_metrics['rec_loss']:.4f} | "
          f"KLD: {val_metrics['kld_loss']:.4f}")
    print(f"        F1-Micro: {val_metrics['f1_micro']:.4f} | "
          f"F1-Macro: {val_metrics['f1_macro']:.4f} | "
          f"Subset Acc: {val_metrics['subset_accuracy']:.4f}")

    # Save best model
    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        patience_counter = 0

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'config': CONFIG
        }
        torch.save(checkpoint, os.path.join(CONFIG['save_dir'], 'best_model.pth'))
        print(f"✓ Best model saved! (Val Loss: {best_val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"Patience: {patience_counter}/{CONFIG['patience']}")

    # Early stopping
    if patience_counter >= CONFIG['patience']:
        print(f"\nEarly stopping triggered after {epoch} epochs")
        break

print(f"\n{'=' * 60}")
print(f"Training completed!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"{'=' * 60}")