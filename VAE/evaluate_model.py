"""
Comprehensive Evaluation Script for EVA Project
Evaluates trained model on test set with detailed metrics
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, hamming_loss,
    multilabel_confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

from model import BetaVAE_SER
from dataset_augmented import EmotionDataset

# Configuration
CHECKPOINT_PATH = 'checkpoints/best_model.pth'
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'evaluation_results'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = BetaVAE_SER(
        n_mels=config['n_mels'],
        n_emotions=config['n_emotions'],
        latent_dim=config['latent_dim']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    print(f"  Val F1-Micro: {checkpoint.get('f1_micro', 'N/A'):.4f}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")

    return model, config


def evaluate_model(model, dataloader, device):
    """
    Evaluate model and collect predictions
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_latents = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc='Evaluating'):
            x = x.to(device)

            # Forward pass
            _, y_pred, mu, _ = model(x)

            # Collect results
            all_preds.append(y_pred.cpu().numpy())
            all_labels.append(y.numpy())
            all_latents.append(mu.cpu().numpy())

    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_latents = np.vstack(all_latents)

    return all_preds, all_labels, all_latents


def calculate_detailed_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate comprehensive metrics for multi-label classification
    """
    y_pred_binary = (y_pred > threshold).astype(int)

    metrics = {
        # Overall metrics
        'subset_accuracy': accuracy_score(y_true, y_pred_binary),
        'hamming_loss': hamming_loss(y_true, y_pred_binary),
        'f1_micro': f1_score(y_true, y_pred_binary, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred_binary, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred_binary, average='weighted', zero_division=0),
        'f1_samples': f1_score(y_true, y_pred_binary, average='samples', zero_division=0),

        # Per-class metrics
        'f1_per_class': f1_score(y_true, y_pred_binary, average=None, zero_division=0),
    }

    # Per-class precision, recall
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred_binary, average=None, zero_division=0
    )

    metrics['precision_per_class'] = precision
    metrics['recall_per_class'] = recall
    metrics['support_per_class'] = support

    return metrics


def plot_per_class_metrics(metrics, save_path):
    """Plot per-class F1, Precision, Recall"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = np.arange(len(EMOTION_LABELS))
    width = 0.6

    # F1 Score
    axes[0].bar(x, metrics['f1_per_class'], width, color='steelblue')
    axes[0].set_xlabel('Emotion')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('F1 Score per Emotion')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(EMOTION_LABELS, rotation=45, ha='right')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(metrics['f1_per_class']):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    # Precision
    axes[1].bar(x, metrics['precision_per_class'], width, color='green', alpha=0.7)
    axes[1].set_xlabel('Emotion')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision per Emotion')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(EMOTION_LABELS, rotation=45, ha='right')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)

    for i, v in enumerate(metrics['precision_per_class']):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    # Recall
    axes[2].bar(x, metrics['recall_per_class'], width, color='orange', alpha=0.7)
    axes[2].set_xlabel('Emotion')
    axes[2].set_ylabel('Recall')
    axes[2].set_title('Recall per Emotion')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(EMOTION_LABELS, rotation=45, ha='right')
    axes[2].set_ylim([0, 1])
    axes[2].grid(axis='y', alpha=0.3)

    for i, v in enumerate(metrics['recall_per_class']):
        axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved per-class metrics plot to {save_path}")


def plot_confusion_matrices(y_true, y_pred, threshold=0.5, save_path='confusion_matrices.png'):
    """Plot confusion matrix for each emotion"""
    y_pred_binary = (y_pred > threshold).astype(int)

    # Get multilabel confusion matrices
    mcm = multilabel_confusion_matrix(y_true, y_pred_binary)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, emotion in enumerate(EMOTION_LABELS):
        cm = mcm[i]

        # Plot heatmap
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            ax=axes[i],
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar=False
        )

        # Calculate accuracy for this emotion
        tn, fp, fn, tp = cm.ravel()
        emotion_acc = (tp + tn) / (tp + tn + fp + fn)

        axes[i].set_title(f'{emotion}\nAccuracy: {emotion_acc:.3f}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrices to {save_path}")


def plot_prediction_distribution(y_pred, save_path):
    """Plot distribution of predicted probabilities"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, emotion in enumerate(EMOTION_LABELS):
        axes[i].hist(y_pred[:, i], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[i].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
        axes[i].set_xlabel('Predicted Probability')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{emotion}')
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved prediction distribution to {save_path}")


def plot_latent_space(latents, labels, save_path):
    """Visualize latent space using t-SNE"""
    from sklearn.manifold import TSNE

    print("Computing t-SNE projection (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_2d = tsne.fit_transform(latents)

    # Create a combined label for visualization (most prominent emotion)
    dominant_emotions = np.argmax(labels, axis=1)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot each emotion class
    for i, emotion in enumerate(EMOTION_LABELS):
        mask = dominant_emotions == i
        ax.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            label=emotion,
            alpha=0.6,
            s=30
        )

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('Latent Space Visualization (t-SNE)')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved latent space visualization to {save_path}")


def generate_classification_report(y_true, y_pred, threshold=0.5, save_path='classification_report.txt'):
    """Generate detailed classification report"""
    y_pred_binary = (y_pred > threshold).astype(int)

    report = classification_report(
        y_true,
        y_pred_binary,
        target_names=EMOTION_LABELS,
        zero_division=0
    )

    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(report)
        f.write("\n")

    print(f"✓ Saved classification report to {save_path}")
    return report


def save_results_summary(metrics, save_path='results_summary.txt'):
    """Save comprehensive results summary"""
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EVALUATION RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")

        f.write("Overall Metrics:\n")
        f.write("-"*70 + "\n")
        f.write(f"  Subset Accuracy (Exact Match): {metrics['subset_accuracy']:.4f}\n")
        f.write(f"  Hamming Loss:                   {metrics['hamming_loss']:.4f}\n")
        f.write(f"  F1-Micro:                       {metrics['f1_micro']:.4f}\n")
        f.write(f"  F1-Macro:                       {metrics['f1_macro']:.4f}\n")
        f.write(f"  F1-Weighted:                    {metrics['f1_weighted']:.4f}\n")
        f.write(f"  F1-Samples:                     {metrics['f1_samples']:.4f}\n")
        f.write("\n")

        f.write("Per-Emotion Metrics:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Emotion':<12} {'F1':<8} {'Precision':<12} {'Recall':<10} {'Support':<10}\n")
        f.write("-"*70 + "\n")

        for i, emotion in enumerate(EMOTION_LABELS):
            f.write(f"{emotion:<12} "
                   f"{metrics['f1_per_class'][i]:<8.4f} "
                   f"{metrics['precision_per_class'][i]:<12.4f} "
                   f"{metrics['recall_per_class'][i]:<10.4f} "
                   f"{int(metrics['support_per_class'][i]):<10}\n")

        f.write("\n")
        f.write("="*70 + "\n")

    print(f"✓ Saved results summary to {save_path}")


def analyze_difficult_samples(y_true, y_pred, top_k=20):
    """Identify most difficult samples (highest error)"""
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculate per-sample error (Hamming distance)
    errors = np.sum(np.abs(y_true - y_pred_binary), axis=1)

    # Get indices of most difficult samples
    difficult_indices = np.argsort(errors)[-top_k:][::-1]

    print(f"\n{'='*70}")
    print(f"TOP {top_k} MOST DIFFICULT SAMPLES")
    print(f"{'='*70}")
    print(f"{'Index':<8} {'Errors':<10} {'True Labels':<30} {'Predicted Labels'}")
    print("-"*70)

    for idx in difficult_indices:
        true_emotions = [EMOTION_LABELS[i] for i in range(8) if y_true[idx, i] == 1]
        pred_emotions = [EMOTION_LABELS[i] for i in range(8) if y_pred_binary[idx, i] == 1]

        print(f"{idx:<8} {errors[idx]:<10} "
              f"{', '.join(true_emotions):<30} "
              f"{', '.join(pred_emotions)}")


def main():
    """Main evaluation pipeline"""
    print("="*70)
    print("EVA PROJECT - MODEL EVALUATION")
    print("="*70)
    print(f"\nDevice: {DEVICE}")

    # Load model
    model, config = load_model(CHECKPOINT_PATH, DEVICE)

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = EmotionDataset(
        audio_dir="EVA_Dataset/processed_audio",
        label_file="EVA_Dataset/labels/test_labels.csv",
        sr=config['sr'],
        n_mels=config['n_mels'],
        duration=config['duration'],
        hop_length=config['hop_length'],
        n_fft=config['n_fft'],
        augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )

    print(f"Test samples: {len(test_dataset)}")

    # Evaluate
    print("\n" + "="*70)
    print("RUNNING EVALUATION")
    print("="*70)

    y_pred, y_true, latents = evaluate_model(model, test_loader, DEVICE)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_detailed_metrics(y_true, y_pred)

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nOverall Performance:")
    print(f"  Subset Accuracy: {metrics['subset_accuracy']:.4f}")
    print(f"  Hamming Loss:    {metrics['hamming_loss']:.4f}")
    print(f"  F1-Micro:        {metrics['f1_micro']:.4f}")
    print(f"  F1-Macro:        {metrics['f1_macro']:.4f}")
    print(f"  F1-Weighted:     {metrics['f1_weighted']:.4f}")

    print(f"\nPer-Emotion F1 Scores:")
    for i, emotion in enumerate(EMOTION_LABELS):
        print(f"  {emotion:<12}: {metrics['f1_per_class'][i]:.4f}")

    # Generate plots
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    plot_per_class_metrics(
        metrics,
        os.path.join(OUTPUT_DIR, 'per_class_metrics.png')
    )

    plot_confusion_matrices(
        y_true, y_pred,
        save_path=os.path.join(OUTPUT_DIR, 'confusion_matrices.png')
    )

    plot_prediction_distribution(
        y_pred,
        save_path=os.path.join(OUTPUT_DIR, 'prediction_distribution.png')
    )

    plot_latent_space(
        latents, y_true,
        save_path=os.path.join(OUTPUT_DIR, 'latent_space_tsne.png')
    )

    # Generate reports
    print("\n" + "="*70)
    print("GENERATING REPORTS")
    print("="*70)

    report = generate_classification_report(
        y_true, y_pred,
        save_path=os.path.join(OUTPUT_DIR, 'classification_report.txt')
    )
    print("\n" + report)

    save_results_summary(
        metrics,
        save_path=os.path.join(OUTPUT_DIR, 'results_summary.txt')
    )

    # Analyze difficult samples
    analyze_difficult_samples(y_true, y_pred, top_k=20)

    # Save predictions for further analysis
    print(f"\n{'='*70}")
    print("SAVING PREDICTIONS")
    print(f"{'='*70}")

    predictions_df = pd.DataFrame(y_pred, columns=EMOTION_LABELS)
    predictions_df.to_csv(
        os.path.join(OUTPUT_DIR, 'predictions.csv'),
        index=False
    )
    print(f"✓ Saved predictions to {OUTPUT_DIR}/predictions.csv")

    labels_df = pd.DataFrame(y_true, columns=EMOTION_LABELS)
    labels_df.to_csv(
        os.path.join(OUTPUT_DIR, 'true_labels.csv'),
        index=False
    )
    print(f"✓ Saved true labels to {OUTPUT_DIR}/true_labels.csv")

    latents_df = pd.DataFrame(latents, columns=[f'latent_{i}' for i in range(latents.shape[1])])
    latents_df.to_csv(
        os.path.join(OUTPUT_DIR, 'latent_representations.csv'),
        index=False
    )
    print(f"✓ Saved latent representations to {OUTPUT_DIR}/latent_representations.csv")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print(f"\nKey files:")
    print(f"  • results_summary.txt - Overall metrics")
    print(f"  • classification_report.txt - Detailed report")
    print(f"  • per_class_metrics.png - Bar charts")
    print(f"  • confusion_matrices.png - Per-emotion confusion matrices")
    print(f"  • latent_space_tsne.png - Latent space visualization")
    print(f"  • predictions.csv - All predictions")


if __name__ == "__main__":
    main()