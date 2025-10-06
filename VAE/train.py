"""
Enhanced Training Script for EVA Project
Uses fixed Beta-VAE model with improvements
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, classification_report
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import fixed model
from model import BetaVAE_SER, beta_vae_loss, BetaScheduler, print_model_summary
from dataset_augmented import EmotionDataset

# --------------------------
# Configuration
# --------------------------
CONFIG = {
    # Model
    'latent_dim': 64,
    'n_mels': 128,
    'n_emotions': 8,

    # Data
    'batch_size': 32,
    'duration': 3,  # seconds
    'sr': 16000,
    'hop_length': 512,
    'n_fft': 2048,

    # Training
    'learning_rate': 1e-3,
    'n_epochs': 100,
    'patience': 15,
    'grad_clip': 1.0,

    # Loss weights
    'alpha': 1.0,  # Classification
    'beta': 0.5,  # KL divergence (with warmup)
    'gamma': 0.1,  # Reconstruction
    'beta_warmup_epochs': 10,

    # Optimizer
    'weight_decay': 1e-5,
    'scheduler': 'cosine',  # 'plateau' or 'cosine'

    # Augmentation
    'use_augmentation': True,

    # Paths
    'save_dir': 'checkpoints',
    'log_dir': 'logs',
    'plot_dir': 'plots',

    # Evaluation
    'eval_every': 1,  # Evaluate every N epochs
    'save_every': 5,  # Save checkpoint every N epochs
}

# Emotion labels
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Create directories
os.makedirs(CONFIG['save_dir'], exist_ok=True)
os.makedirs(CONFIG['log_dir'], exist_ok=True)
os.makedirs(CONFIG['plot_dir'], exist_ok=True)

# --------------------------
# Device Setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# --------------------------
# Dataset & DataLoader
# --------------------------
print("\n" + "=" * 70)
print("LOADING DATASETS")
print("=" * 70)

train_dataset = EmotionDataset(
    audio_dir="EVA_Dataset/processed_audio",
    label_file="EVA_Dataset/labels/train_labels.csv",
    sr=CONFIG['sr'],
    n_mels=CONFIG['n_mels'],
    duration=CONFIG['duration'],
    hop_length=CONFIG['hop_length'],
    n_fft=CONFIG['n_fft'],
    augment=CONFIG['use_augmentation']
)

val_dataset = EmotionDataset(
    audio_dir="EVA_Dataset/processed_audio",
    label_file="EVA_Dataset/labels/val_labels.csv",
    sr=CONFIG['sr'],
    n_mels=CONFIG['n_mels'],
    duration=CONFIG['duration'],
    hop_length=CONFIG['hop_length'],
    n_fft=CONFIG['n_fft'],
    augment=False  # No augmentation for validation
)

test_dataset = EmotionDataset(
    audio_dir="EVA_Dataset/processed_audio",
    label_file="EVA_Dataset/labels/test_labels.csv",
    sr=CONFIG['sr'],
    n_mels=CONFIG['n_mels'],
    duration=CONFIG['duration'],
    hop_length=CONFIG['hop_length'],
    n_fft=CONFIG['n_fft'],
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=2,
    pin_memory=(device.type == 'cuda'),
    drop_last=True  # For batch norm stability
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    num_workers=2,
    pin_memory=(device.type == 'cuda')
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['batch_size'],
    num_workers=2,
    pin_memory=(device.type == 'cuda')
)

print(f"\nTrain samples: {len(train_dataset)}")
print(f"Val samples:   {len(val_dataset)}")
print(f"Test samples:  {len(test_dataset)}")
print(f"Batch size:    {CONFIG['batch_size']}")
print(f"Augmentation:  {'ON' if CONFIG['use_augmentation'] else 'OFF'}")

# --------------------------
# Model Initialization
# --------------------------
print("\n" + "=" * 70)
print("INITIALIZING MODEL")
print("=" * 70)

model = BetaVAE_SER(
    n_mels=CONFIG['n_mels'],
    n_emotions=CONFIG['n_emotions'],
    latent_dim=CONFIG['latent_dim']
).to(device)

print_model_summary(model, input_shape=(1, 1, CONFIG['n_mels'], 94))

# --------------------------
# Optimizer & Scheduler
# --------------------------
optimizer = optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)

if CONFIG['scheduler'] == 'plateau':
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=7,
        verbose=True
    )
elif CONFIG['scheduler'] == 'cosine':
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the period after each restart
        eta_min=1e-6
    )

beta_scheduler = BetaScheduler(
    n_epochs=CONFIG['n_epochs'],
    target_beta=CONFIG['beta'],
    warmup_epochs=CONFIG['beta_warmup_epochs']
)


# --------------------------
# Metrics Calculation
# --------------------------
def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Multi-label classification metrics
    """
    y_pred_binary = (y_pred > threshold).astype(int)
    y_true_np = y_true if isinstance(y_true, np.ndarray) else y_true.cpu().numpy()

    # Subset accuracy (exact match)
    subset_acc = accuracy_score(y_true_np, y_pred_binary)

    # Hamming loss
    hamming = hamming_loss(y_true_np, y_pred_binary)

    # F1 scores
    f1_micro = f1_score(y_true_np, y_pred_binary, average='micro', zero_division=0)
    f1_macro = f1_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
    f1_samples = f1_score(y_true_np, y_pred_binary, average='samples', zero_division=0)

    # Per-class F1
    f1_per_class = f1_score(y_true_np, y_pred_binary, average=None, zero_division=0)

    return {
        'subset_accuracy': subset_acc,
        'hamming_loss': hamming,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_samples': f1_samples,
        'f1_per_class': f1_per_class
    }


# --------------------------
# Training Function
# --------------------------
def train_epoch(model, loader, optimizer, device, config, beta_scheduler):
    model.train()

    total_loss = 0
    total_cls = 0
    total_rec = 0
    total_kld = 0

    all_preds = []
    all_labels = []

    # Get current beta
    beta = beta_scheduler.step()

    pbar = tqdm(loader, desc='Training', ncols=100)
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward pass
        x_recon, y_pred, mu, log_var = model(x)

        # Calculate loss with beta warmup
        loss, cls, rec, kld, loss_dict = beta_vae_loss(
            x, x_recon, y, y_pred, mu, log_var,
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma'],
            warmup_beta=beta
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_cls += cls.item()
        total_rec += rec.item()
        total_kld += kld.item()

        # Collect predictions
        all_preds.append(y_pred.detach().cpu().numpy())
        all_labels.append(y.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'β': f'{beta:.3f}'
        })

    n = len(loader)

    # Calculate metrics
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = calculate_metrics(all_labels, all_preds)

    return {
        'loss': total_loss / n,
        'cls_loss': total_cls / n,
        'rec_loss': total_rec / n,
        'kld_loss': total_kld / n,
        'beta_used': beta,
        **metrics
    }


# --------------------------
# Validation Function
# --------------------------
def eval_epoch(model, loader, device, config, beta=None):
    model.eval()

    total_loss = 0
    total_cls = 0
    total_rec = 0
    total_kld = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc='Validation', ncols=100):
            x, y = x.to(device), y.to(device)

            x_recon, y_pred, mu, log_var = model(x)

            loss, cls, rec, kld, loss_dict = beta_vae_loss(
                x, x_recon, y, y_pred, mu, log_var,
                alpha=config['alpha'],
                beta=config['beta'],
                gamma=config['gamma'],
                warmup_beta=beta
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
    metrics = calculate_metrics(all_labels, all_preds)

    return {
        'loss': total_loss / n,
        'cls_loss': total_cls / n,
        'rec_loss': total_rec / n,
        'kld_loss': total_kld / n,
        **metrics
    }


# --------------------------
# Plotting Functions
# --------------------------
def plot_training_curves(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # F1 scores
    axes[0, 1].plot(history['train_f1_micro'], label='Train Micro')
    axes[0, 1].plot(history['val_f1_micro'], label='Val Micro')
    axes[0, 1].plot(history['train_f1_macro'], label='Train Macro')
    axes[0, 1].plot(history['val_f1_macro'], label='Val Macro')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Scores')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Loss components
    axes[1, 0].plot(history['train_cls_loss'], label='Classification')
    axes[1, 0].plot(history['train_rec_loss'], label='Reconstruction')
    axes[1, 0].plot(history['train_kld_loss'], label='KL Divergence')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss Components')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Beta schedule
    if 'beta_used' in history:
        axes[1, 1].plot(history['beta_used'], label='β')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Beta Value')
        axes[1, 1].set_title('Beta Warmup Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix_per_emotion(y_true, y_pred, save_path):
    """Plot per-emotion performance"""
    y_pred_binary = (y_pred > 0.5).astype(int)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, emotion in enumerate(EMOTION_LABELS):
        # Confusion matrix for each emotion
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])

        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'{emotion}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# --------------------------
# Training Loop
# --------------------------
def train():
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key:20s}: {value}")
    print("=" * 70)

    best_val_loss = float('inf')
    best_f1_micro = 0.0
    patience_counter = 0

    # History tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_cls_loss': [], 'val_cls_loss': [],
        'train_rec_loss': [], 'val_rec_loss': [],
        'train_kld_loss': [], 'val_kld_loss': [],
        'train_f1_micro': [], 'val_f1_micro': [],
        'train_f1_macro': [], 'val_f1_macro': [],
        'beta_used': []
    }

    start_time = datetime.now()

    for epoch in range(1, CONFIG['n_epochs'] + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{CONFIG['n_epochs']}")
        print(f"{'=' * 70}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, CONFIG, beta_scheduler
        )

        # Validate
        if epoch % CONFIG['eval_every'] == 0:
            val_metrics = eval_epoch(
                model, val_loader, device, CONFIG,
                beta=train_metrics['beta_used']
            )
        else:
            val_metrics = None

        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        if CONFIG['scheduler'] == 'plateau' and val_metrics:
            lr_scheduler.step(val_metrics['loss'])
        elif CONFIG['scheduler'] == 'cosine':
            lr_scheduler.step()

        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"\n📉 Learning rate: {current_lr:.6f} → {new_lr:.6f}")

        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f} | "
              f"Cls: {train_metrics['cls_loss']:.4f} | "
              f"Rec: {train_metrics['rec_loss']:.4f} | "
              f"KLD: {train_metrics['kld_loss']:.4f}")
        print(f"        F1-Micro: {train_metrics['f1_micro']:.4f} | "
              f"F1-Macro: {train_metrics['f1_macro']:.4f} | "
              f"β: {train_metrics['beta_used']:.4f}")

        if val_metrics:
            print(f"\nVal   - Loss: {val_metrics['loss']:.4f} | "
                  f"Cls: {val_metrics['cls_loss']:.4f} | "
                  f"Rec: {val_metrics['rec_loss']:.4f} | "
                  f"KLD: {val_metrics['kld_loss']:.4f}")
            print(f"        F1-Micro: {val_metrics['f1_micro']:.4f} | "
                  f"F1-Macro: {val_metrics['f1_macro']:.4f}")

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_cls_loss'].append(train_metrics['cls_loss'])
        history['train_rec_loss'].append(train_metrics['rec_loss'])
        history['train_kld_loss'].append(train_metrics['kld_loss'])
        history['train_f1_micro'].append(train_metrics['f1_micro'])
        history['train_f1_macro'].append(train_metrics['f1_macro'])
        history['beta_used'].append(train_metrics['beta_used'])

        if val_metrics:
            history['val_loss'].append(val_metrics['loss'])
            history['val_cls_loss'].append(val_metrics['cls_loss'])
            history['val_rec_loss'].append(val_metrics['rec_loss'])
            history['val_kld_loss'].append(val_metrics['kld_loss'])
            history['val_f1_micro'].append(val_metrics['f1_micro'])
            history['val_f1_macro'].append(val_metrics['f1_macro'])

        # Save best model
        if val_metrics and val_metrics['f1_micro'] > best_f1_micro:
            best_f1_micro = val_metrics['f1_micro']
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'val_loss': val_metrics['loss'],
                'f1_micro': val_metrics['f1_micro'],
                'config': CONFIG,
                'history': history
            }
            torch.save(checkpoint, os.path.join(CONFIG['save_dir'], 'best_model.pth'))
            print(f"\n✓ Best model saved! (F1: {best_f1_micro:.4f}, Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"\nPatience: {patience_counter}/{CONFIG['patience']}")

        # Save periodic checkpoint
        if epoch % CONFIG['save_every'] == 0:
            torch.save(checkpoint, os.path.join(CONFIG['save_dir'], f'checkpoint_epoch_{epoch}.pth'))

        # Early stopping
        if patience_counter >= CONFIG['patience']:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs")
            break

        # Plot curves
        if epoch % 5 == 0:
            plot_training_curves(
                history,
                os.path.join(CONFIG['plot_dir'], f'training_curves_epoch_{epoch}.png')
            )

    # Training complete
    elapsed_time = datetime.now() - start_time
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Total time: {elapsed_time}")
    print(f"Best F1-Micro: {best_f1_micro:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")

    # Final plots
    plot_training_curves(
        history,
        os.path.join(CONFIG['plot_dir'], 'final_training_curves.png')
    )

    # Save history
    with open(os.path.join(CONFIG['log_dir'], 'training_history.json'), 'w') as f:
        # Convert numpy to list for JSON serialization
        history_json = {k: [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
                        for k, v in history.items()}
        json.dump(history_json, f, indent=2)

    return history


if __name__ == "__main__":
    history = train()

    print("\n🚀 Next steps:")
    print("   1. Evaluate on test set: python VAE/evaluate.py")
    print("   2. Test inference: python VAE/inference.py")
    print("   3. View plots: ls plots/")