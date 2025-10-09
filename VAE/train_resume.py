"""
Resume Training Script for EVA Project
Continues training from an existing checkpoint with full state restoration
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

# Import model and dataset
from model import BetaVAE_SER, beta_vae_loss, BetaScheduler, print_model_summary
from dataset_augmented import EmotionDataset


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}\n")


# --------------------------
# Configuration
# --------------------------
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


# --------------------------
# Load Checkpoint
# --------------------------
def load_checkpoint(checkpoint_path, device):
    """
    Load checkpoint and restore training state

    Returns:
        model, optimizer, scheduler, config, history, start_epoch, best_metrics
    """
    print_header("LOADING CHECKPOINT")

    if not os.path.exists(checkpoint_path):
        print_error(f"Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print_info(f"Loading from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract checkpoint info
    config = checkpoint['config']
    history = checkpoint.get('history', {})
    start_epoch = checkpoint['epoch'] + 1  # Start from next epoch

    print_success(f"Checkpoint from epoch {checkpoint['epoch']}")
    print_info(f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print_info(f"F1-Micro: {checkpoint.get('f1_micro', 'N/A'):.4f}")

    # Initialize model
    print_info("\nInitializing model...")
    model = BetaVAE_SER(
        n_mels=config['n_mels'],
        n_emotions=config['n_emotions'],
        latent_dim=config['latent_dim']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print_success("Model state loaded")

    # Initialize optimizer
    print_info("Initializing optimizer...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5)
    )

    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print_success("Optimizer state loaded")
    else:
        print_info("No optimizer state found, using fresh optimizer")

    # Initialize scheduler
    print_info("Initializing scheduler...")
    scheduler_type = config.get('scheduler', 'cosine')

    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=7,
            verbose=True
        )
    else:  # cosine
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

    if 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print_success("Scheduler state loaded")
        except:
            print_info("Could not load scheduler state, using fresh scheduler")
    else:
        print_info("No scheduler state found, using fresh scheduler")

    # Best metrics
    best_metrics = {
        'val_loss': checkpoint.get('val_loss', float('inf')),
        'f1_micro': checkpoint.get('f1_micro', 0.0),
        'patience_counter': 0
    }

    # Validate history structure
    if not history:
        print_info("No training history found, starting fresh history")
        history = {
            'train_loss': [], 'val_loss': [],
            'train_cls_loss': [], 'val_cls_loss': [],
            'train_rec_loss': [], 'val_rec_loss': [],
            'train_kld_loss': [], 'val_kld_loss': [],
            'train_f1_micro': [], 'val_f1_micro': [],
            'train_f1_macro': [], 'val_f1_macro': [],
            'beta_used': []
        }

    print_success(f"Will resume from epoch {start_epoch}")

    return model, optimizer, scheduler, config, history, start_epoch, best_metrics


# --------------------------
# Metrics Calculation
# --------------------------
def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Multi-label classification metrics"""
    y_pred_binary = (y_pred > threshold).astype(int)
    y_true_np = y_true if isinstance(y_true, np.ndarray) else y_true.cpu().numpy()

    subset_acc = accuracy_score(y_true_np, y_pred_binary)
    hamming = hamming_loss(y_true_np, y_pred_binary)
    f1_micro = f1_score(y_true_np, y_pred_binary, average='micro', zero_division=0)
    f1_macro = f1_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
    f1_samples = f1_score(y_true_np, y_pred_binary, average='samples', zero_division=0)
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

    beta = beta_scheduler.step()

    pbar = tqdm(loader, desc='Training', ncols=100)
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        x_recon, y_pred, mu, log_var = model(x)

        loss, cls, rec, kld, loss_dict = beta_vae_loss(
            x, x_recon, y, y_pred, mu, log_var,
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma'],
            warmup_beta=beta
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()

        total_loss += loss.item()
        total_cls += cls.item()
        total_rec += rec.item()
        total_kld += kld.item()

        all_preds.append(y_pred.detach().cpu().numpy())
        all_labels.append(y.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'β': f'{beta:.3f}'})

    n = len(loader)
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

    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_f1_micro'], label='Train Micro')
    axes[0, 1].plot(history['val_f1_micro'], label='Val Micro')
    axes[0, 1].plot(history['train_f1_macro'], label='Train Macro')
    axes[0, 1].plot(history['val_f1_macro'], label='Val Macro')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Scores')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history['train_cls_loss'], label='Classification')
    axes[1, 0].plot(history['train_rec_loss'], label='Reconstruction')
    axes[1, 0].plot(history['train_kld_loss'], label='KL Divergence')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss Components')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    if 'beta_used' in history and history['beta_used']:
        axes[1, 1].plot(history['beta_used'], label='β')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Beta Value')
        axes[1, 1].set_title('Beta Warmup Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# --------------------------
# Main Training Loop
# --------------------------
def resume_training(checkpoint_path, additional_epochs=None, new_patience=None,
                    dataset_path="EVA_Dataset", kaggle_mode=False):
    """
    Resume training from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        additional_epochs: Number of additional epochs to train (None = use config)
        new_patience: New patience value (None = use config)
        dataset_path: Path to dataset directory
        kaggle_mode: If True, use Kaggle paths
    """

    print_header("EVA PROJECT - RESUME TRAINING")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_info(f"Using device: {device}")
    if torch.cuda.is_available():
        print_info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load checkpoint
    model, optimizer, scheduler, config, history, start_epoch, best_metrics = \
        load_checkpoint(checkpoint_path, device)

    # Override config if specified
    if additional_epochs is not None:
        original_epochs = config['n_epochs']
        config['n_epochs'] = start_epoch + additional_epochs - 1
        print_info(f"\nExtending training: {original_epochs} → {config['n_epochs']} epochs")

    if new_patience is not None:
        config['patience'] = new_patience
        print_info(f"New patience: {new_patience}")

    # Adjust paths for Kaggle
    if kaggle_mode:
        audio_dir = f"/kaggle/working/{dataset_path}/processed_audio"
        train_labels = f"/kaggle/working/{dataset_path}/labels/train_labels.csv"
        val_labels = f"/kaggle/working/{dataset_path}/labels/val_labels.csv"
    else:
        audio_dir = f"{dataset_path}/processed_audio"
        train_labels = f"{dataset_path}/labels/train_labels.csv"
        val_labels = f"{dataset_path}/labels/val_labels.csv"

    # Load datasets
    print_header("LOADING DATASETS")

    train_dataset = EmotionDataset(
        audio_dir=audio_dir,
        label_file=train_labels,
        sr=config['sr'],
        n_mels=config['n_mels'],
        duration=config['duration'],
        hop_length=config['hop_length'],
        n_fft=config['n_fft'],
        augment=config.get('use_augmentation', True)
    )

    val_dataset = EmotionDataset(
        audio_dir=audio_dir,
        label_file=val_labels,
        sr=config['sr'],
        n_mels=config['n_mels'],
        duration=config['duration'],
        hop_length=config['hop_length'],
        n_fft=config['n_fft'],
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=2,
        pin_memory=(device.type == 'cuda')
    )

    print_success(f"Train samples: {len(train_dataset)}")
    print_success(f"Val samples: {len(val_dataset)}")

    # Initialize beta scheduler from current epoch
    beta_scheduler = BetaScheduler(
        n_epochs=config['n_epochs'],
        target_beta=config.get('beta', 0.5),
        warmup_epochs=config.get('beta_warmup_epochs', 10)
    )
    # Fast-forward beta scheduler to current epoch
    beta_scheduler.current_epoch = start_epoch - 1

    # Create directories
    save_dir = config.get('save_dir', 'checkpoints')
    log_dir = config.get('log_dir', 'logs')
    plot_dir = config.get('plot_dir', 'plots')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Training loop
    print_header("RESUMING TRAINING")
    print_info(f"Epochs: {start_epoch} → {config['n_epochs']}")
    print_info(f"Best F1-Micro so far: {best_metrics['f1_micro']:.4f}")
    print_info(f"Best Val Loss so far: {best_metrics['val_loss']:.4f}")

    patience_counter = best_metrics['patience_counter']
    best_f1_micro = best_metrics['f1_micro']
    best_val_loss = best_metrics['val_loss']

    start_time = datetime.now()

    for epoch in range(start_epoch, config['n_epochs'] + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{config['n_epochs']}")
        print(f"{'=' * 70}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, config, beta_scheduler
        )

        # Validate
        if epoch % config.get('eval_every', 1) == 0:
            val_metrics = eval_epoch(
                model, val_loader, device, config,
                beta=train_metrics['beta_used']
            )
        else:
            val_metrics = None

        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        if config['scheduler'] == 'plateau' and val_metrics:
            scheduler.step(val_metrics['loss'])
        elif config['scheduler'] == 'cosine':
            scheduler.step()

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
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['loss'],
                'f1_micro': val_metrics['f1_micro'],
                'config': config,
                'history': history
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print_success(f"Best model saved! (F1: {best_f1_micro:.4f}, Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print_info(f"Patience: {patience_counter}/{config['patience']}")

        # Save periodic checkpoint
        if epoch % config.get('save_every', 5) == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['loss'] if val_metrics else float('inf'),
                'f1_micro': val_metrics['f1_micro'] if val_metrics else 0.0,
                'config': config,
                'history': history
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
            print_success(f"Checkpoint saved: epoch_{epoch}.pth")

        # Early stopping
        if patience_counter >= config['patience']:
            print_info(f"\n⚠ Early stopping triggered after {epoch} epochs")
            break

        # Plot curves
        if epoch % 5 == 0:
            plot_training_curves(
                history,
                os.path.join(plot_dir, f'training_curves_epoch_{epoch}.png')
            )

    # Training complete
    elapsed_time = datetime.now() - start_time

    print_header("TRAINING COMPLETE")
    print_success(f"Total time: {elapsed_time}")
    print_success(f"Best F1-Micro: {best_f1_micro:.4f}")
    print_success(f"Best Val Loss: {best_val_loss:.4f}")

    # Final plots
    plot_training_curves(
        history,
        os.path.join(plot_dir, 'final_training_curves.png')
    )

    # Save history
    with open(os.path.join(log_dir, 'training_history.json'), 'w') as f:
        history_json = {k: [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
                        for k, v in history.items()}
        json.dump(history_json, f, indent=2)

    print_success(f"\nAll results saved to: {save_dir}/")

    return history


# --------------------------
# CLI Interface
# --------------------------
def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('checkpoint', type=str,
                        help='Path to checkpoint file (e.g., checkpoints/best_model.pth)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Additional epochs to train (default: use config)')
    parser.add_argument('--patience', type=int, default=None,
                        help='New patience value (default: use config)')
    parser.add_argument('--dataset', type=str, default='EVA_Dataset',
                        help='Path to dataset directory')
    parser.add_argument('--kaggle', action='store_true',
                        help='Use Kaggle paths')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print_error(f"Checkpoint not found: {args.checkpoint}")
        print_info("\nAvailable checkpoints:")
        if os.path.exists('checkpoints'):
            for f in os.listdir('checkpoints'):
                if f.endswith('.pth'):
                    print(f"  - checkpoints/{f}")
        return

    resume_training(
        checkpoint_path=args.checkpoint,
        additional_epochs=args.epochs,
        new_patience=args.patience,
        dataset_path=args.dataset,
        kaggle_mode=args.kaggle
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        print("Resume Training Script")
        print("=" * 70)
        print("\nUsage:")
        print("  python train_resume.py <checkpoint_path> [options]")
        print("\nExamples:")
        print("  # Resume from best model, train 20 more epochs")
        print("  python train_resume.py checkpoints/best_model.pth --epochs 20")
        print("\n  # Resume with new patience")
        print("  python train_resume.py checkpoints/checkpoint_epoch_50.pth --patience 10")
        print("\n  # Resume on Kaggle")
        print("  python train_resume.py checkpoints/best_model.pth --kaggle --epochs 30")
        print("\nOptions:")
        print("  --epochs N     Train for N additional epochs")
        print("  --patience N   Use new patience value")
        print("  --dataset PATH Custom dataset path")
        print("  --kaggle       Use Kaggle paths (/kaggle/working/...)")