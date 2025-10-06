"""
Evaluate trained model on test set
Get detailed metrics and confusion matrix
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score
from model_fixed import BetaVAE_SER
from dataset import EmotionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# --------------------------
# Configuration
# --------------------------
CHECKPOINT_PATH = 'checkpoints/best_model.pth'
TEST_DATA_DIR = 'EVA_Dataset/processed_audio'
TEST_LABEL_FILE = 'EVA_Dataset/labels/test_labels.csv'

EMOTION_LABELS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']


# --------------------------
# Load Model
# --------------------------
def load_model(checkpoint_path, device='cuda'):
    """Load trained model"""
    print(f"📥 Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = BetaVAE_SER(
        n_mels=128,
        n_emotions=8,
        latent_dim=checkpoint['config']['latent_dim']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Model loaded (trained for {checkpoint['epoch']} epochs)")
    print(f"   Best val loss: {checkpoint['val_loss']:.4f}")

    return model


# --------------------------
# Evaluate on Test Set
# --------------------------
def evaluate(model, test_loader, device, threshold=0.5):
    """Evaluate model and return predictions"""
    print(f"\n🧪 Evaluating on test set...")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)

            _, y_pred, _, _ = model(x)

            all_probs.append(y_pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    # Concatenate all batches
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    all_preds = (all_probs > threshold).astype(int)

    return all_labels, all_preds, all_probs


# --------------------------
# Calculate Metrics
# --------------------------
def calculate_detailed_metrics(y_true, y_pred, y_probs):
    """Calculate comprehensive metrics"""
    print("\n" + "=" * 70)
    print("📊 EVALUATION METRICS")
    print("=" * 70)

    metrics = {}

    # Overall metrics
    metrics['subset_accuracy'] = accuracy_score(y_true, y_pred)
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    metrics['jaccard_score'] = jaccard_score(y_true, y_pred, average='samples')

    # F1 scores
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_samples'] = f1_score(y_true, y_pred, average='samples', zero_division=0)

    print(f"\n📈 Overall Performance:")
    print(f"  Subset Accuracy:  {metrics['subset_accuracy']:.4f} (exact match)")
    print(f"  Hamming Loss:     {metrics['hamming_loss']:.4f} (lower is better)")
    print(f"  Jaccard Score:    {metrics['jaccard_score']:.4f} (IoU)")
    print(f"\n  F1-Micro:         {metrics['f1_micro']:.4f}")
    print(f"  F1-Macro:         {metrics['f1_macro']:.4f}")
    print(f"  F1-Weighted:      {metrics['f1_weighted']:.4f}")
    print(f"  F1-Samples:       {metrics['f1_samples']:.4f}")

    # Per-emotion metrics
    print(f"\n📊 Per-Emotion Performance:")
    print(f"{'Emotion':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 60)

    per_emotion_metrics = []

    for i, emotion in enumerate(EMOTION_LABELS):
        y_true_emotion = y_true[:, i]
        y_pred_emotion = y_pred[:, i]

        # Calculate metrics for this emotion
        from sklearn.metrics import precision_score, recall_score

        precision = precision_score(y_true_emotion, y_pred_emotion, zero_division=0)
        recall = recall_score(y_true_emotion, y_pred_emotion, zero_division=0)
        f1 = f1_score(y_true_emotion, y_pred_emotion, zero_division=0)
        support = int(y_true_emotion.sum())

        per_emotion_metrics.append({
            'emotion': emotion,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        })

        print(f"{emotion:<12} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10d}")

    metrics['per_emotion'] = per_emotion_metrics

    return metrics


# --------------------------
# Visualization
# --------------------------
def plot_confusion_matrices(y_true, y_pred):
    """Plot confusion matrix for each emotion"""
    print("\n📊 Generating confusion matrices...")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, emotion in enumerate(EMOTION_LABELS):
        y_true_emotion = y_true[:, i]
        y_pred_emotion = y_pred[:, i]

        cm = confusion_matrix(y_true_emotion, y_pred_emotion)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        axes[i].set_title(f'{emotion}')
        axes[i].set_ylabel('True')
        axes[i].set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: confusion_matrices.png")
    plt.close()


def plot_emotion_distribution(y_true, y_pred):
    """Compare true vs predicted emotion distribution"""
    print("\n📊 Generating emotion distribution plot...")

    true_counts = y_true.sum(axis=0)
    pred_counts = y_pred.sum(axis=0)

    x = np.arange(len(EMOTION_LABELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, true_counts, width, label='True', alpha=0.8)
    ax.bar(x + width / 2, pred_counts, width, label='Predicted', alpha=0.8)

    ax.set_xlabel('Emotions')
    ax.set_ylabel('Count')
    ax.set_title('Emotion Distribution: True vs Predicted')
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('emotion_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: emotion_distribution.png")
    plt.close()


def plot_f1_scores(metrics):
    """Plot F1 scores per emotion"""
    print("\n📊 Generating F1 score plot...")

    emotions = [m['emotion'] for m in metrics['per_emotion']]
    f1_scores = [m['f1'] for m in metrics['per_emotion']]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(emotions, f1_scores, color='steelblue', alpha=0.8)

    # Color code by performance
    for i, bar in enumerate(bars):
        if f1_scores[i] >= 0.7:
            bar.set_color('green')
        elif f1_scores[i] >= 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    ax.axhline(y=metrics['f1_macro'], color='red', linestyle='--',
               label=f"Macro F1: {metrics['f1_macro']:.3f}")
    ax.set_xlabel('Emotions')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Scores by Emotion')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('f1_scores.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: f1_scores.png")
    plt.close()


# --------------------------
# Main Evaluation
# --------------------------
def main():
    print("=" * 70)
    print("🎯 EVA MODEL EVALUATION")
    print("=" * 70)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_model(CHECKPOINT_PATH, device)

    # Load test dataset
    print(f"\n📂 Loading test dataset...")
    test_dataset = EmotionDataset(
        audio_dir=TEST_DATA_DIR,
        label_file=TEST_LABEL_FILE
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"   Test samples: {len(test_dataset)}")

    # Evaluate
    y_true, y_pred, y_probs = evaluate(model, test_loader, device)

    # Calculate metrics
    metrics = calculate_detailed_metrics(y_true, y_pred, y_probs)

    # Generate visualizations
    plot_confusion_matrices(y_true, y_pred)
    plot_emotion_distribution(y_true, y_pred)
    plot_f1_scores(metrics)

    # Save results
    print(f"\n💾 Saving evaluation results...")

    results = {
        'overall_metrics': {
            'subset_accuracy': float(metrics['subset_accuracy']),
            'hamming_loss': float(metrics['hamming_loss']),
            'jaccard_score': float(metrics['jaccard_score']),
            'f1_micro': float(metrics['f1_micro']),
            'f1_macro': float(metrics['f1_macro']),
            'f1_weighted': float(metrics['f1_weighted']),
            'f1_samples': float(metrics['f1_samples'])
        },
        'per_emotion_metrics': metrics['per_emotion'],
        'test_samples': len(test_dataset)
    }

    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("✅ Saved: evaluation_results.json")

    # Summary
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   F1-Macro:  {metrics['f1_macro']:.4f}")
    print(f"   F1-Micro:  {metrics['f1_micro']:.4f}")
    print(f"   Accuracy:  {metrics['subset_accuracy']:.4f}")

    print(f"\n📁 Generated files:")
    print(f"   - confusion_matrices.png")
    print(f"   - emotion_distribution.png")
    print(f"   - f1_scores.png")
    print(f"   - evaluation_results.json")

    print(f"\n🚀 Next steps:")
    print(f"   1. Review the visualizations")
    print(f"   2. Test on real audio: python inference.py")
    print(f"   3. Fine-tune on Vietnamese data (if needed)")


if __name__ == "__main__":
    main()