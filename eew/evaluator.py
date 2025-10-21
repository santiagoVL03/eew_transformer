"""
Evaluator module for Transformer model.

Features:
- Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- MC Dropout uncertainty quantification
- Visualization (ROC curves, confusion matrix, uncertainty plots)
"""

import logging
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class Evaluator:
    """
    Evaluator for Transformer earthquake detection model.
    """
    
    def __init__(self, model, test_loader, device='cuda', threshold=0.5):
        """
        Args:
            model: Transformer model
            test_loader: Test data loader
            device: Device to run evaluation on
            threshold: Classification threshold (default 0.5)
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.threshold = threshold
        
        self.predictions = None
        self.probabilities = None
        self.labels_true = None
        self.uncertainties = None
    
    def evaluate(self, use_mc_dropout=False, n_passes=10):
        """
        Evaluate model on test set.
        
        Args:
            use_mc_dropout: Use MC Dropout for uncertainty estimation
            n_passes: Number of MC dropout passes
        
        Returns:
            Dictionary of metrics
        """
        logging.info("Evaluating model on test set...")
        
        self.model.eval()
        
        all_probs = []
        all_labels = []
        all_uncertainties = []
        
        with torch.no_grad():
            for waveforms, labels in self.test_loader:
                waveforms = waveforms.to(self.device)
                labels = labels.cpu().numpy()
                
                if use_mc_dropout:
                    # MC Dropout prediction
                    mean_prob, std_prob, entropy = self.model.predict_with_uncertainty(
                        waveforms, n_passes=n_passes
                    )
                    probs = mean_prob.cpu().numpy()
                    uncertainties = entropy.cpu().numpy()
                else:
                    # Standard prediction
                    logits = self.model(waveforms)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    uncertainties = np.zeros_like(probs)  # No uncertainty
                
                all_probs.extend(probs)
                all_labels.extend(labels.flatten())
                all_uncertainties.extend(uncertainties)
        
        # Convert to arrays
        self.probabilities = np.array(all_probs)
        self.labels_true = np.array(all_labels)
        self.predictions = (self.probabilities > self.threshold).astype(int)
        self.uncertainties = np.array(all_uncertainties)
        
        # Compute metrics
        metrics = self.compute_metrics()
        
        logging.info("Evaluation completed!")
        self.log_metrics(metrics)
        
        return metrics
    
    def compute_metrics(self):
        """
        Compute classification metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(self.labels_true, self.predictions),
            'precision': precision_score(self.labels_true, self.predictions, zero_division=0),
            'recall': recall_score(self.labels_true, self.predictions, zero_division=0),
            'f1': f1_score(self.labels_true, self.predictions, zero_division=0),
        }
        
        # ROC-AUC (only if both classes present)
        if len(np.unique(self.labels_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(self.labels_true, self.probabilities)
        else:
            metrics['roc_auc'] = np.nan
        
        # Uncertainty metrics
        if self.uncertainties is not None:
            metrics['mean_uncertainty'] = np.mean(self.uncertainties)
            metrics['std_uncertainty'] = np.std(self.uncertainties)
        
        return metrics
    
    def log_metrics(self, metrics):
        """Log metrics to console."""
        logging.info("\n" + "=" * 50)
        logging.info("Test Set Metrics:")
        logging.info("=" * 50)
        logging.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logging.info(f"Precision: {metrics['precision']:.4f}")
        logging.info(f"Recall:    {metrics['recall']:.4f}")
        logging.info(f"F1 Score:  {metrics['f1']:.4f}")
        logging.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        if 'mean_uncertainty' in metrics:
            logging.info(f"Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")
            logging.info(f"Std Uncertainty:  {metrics['std_uncertainty']:.4f}")
        
        logging.info("=" * 50 + "\n")
        
        # Classification report
        logging.info("Classification Report:")
        print(classification_report(
            self.labels_true,
            self.predictions,
            target_names=['Noise', 'Earthquake'],
            zero_division=0
        ))
    
    def plot_roc_curve(self, save_path=None):
        """
        Plot ROC curve.
        
        Args:
            save_path: Path to save plot (if None, only display)
        """
        if len(np.unique(self.labels_true)) < 2:
            logging.warning("Cannot plot ROC curve: only one class present in labels")
            return
        
        fpr, tpr, thresholds = roc_curve(self.labels_true, self.probabilities)
        auc = roc_auc_score(self.labels_true, self.probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'Transformer (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Transformer Earthquake Detection', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"ROC curve saved: {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            save_path: Path to save plot (if None, only display)
        """
        cm = confusion_matrix(self.labels_true, self.predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Noise', 'Earthquake'],
            yticklabels=['Noise', 'Earthquake'],
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix - Transformer', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Confusion matrix saved: {save_path}")
        
        plt.close()
    
    def plot_uncertainty_analysis(self, save_path=None):
        """
        Plot uncertainty analysis.
        
        Shows:
        - Uncertainty distribution for correct vs incorrect predictions
        - Uncertainty vs confidence
        
        Args:
            save_path: Path to save plot (if None, only display)
        """
        if self.uncertainties is None or np.all(self.uncertainties == 0):
            logging.warning("No uncertainty data available")
            return
        
        correct = (self.predictions == self.labels_true)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Uncertainty distribution
        axes[0].hist(
            self.uncertainties[correct],
            bins=30,
            alpha=0.6,
            label='Correct',
            color='green'
        )
        axes[0].hist(
            self.uncertainties[~correct],
            bins=30,
            alpha=0.6,
            label='Incorrect',
            color='red'
        )
        axes[0].set_xlabel('Uncertainty (Entropy)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Uncertainty Distribution', fontsize=13)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Uncertainty vs Confidence
        confidence = np.maximum(self.probabilities, 1 - self.probabilities)
        
        axes[1].scatter(
            confidence[correct],
            self.uncertainties[correct],
            alpha=0.3,
            s=10,
            label='Correct',
            color='green'
        )
        axes[1].scatter(
            confidence[~correct],
            self.uncertainties[~correct],
            alpha=0.5,
            s=20,
            label='Incorrect',
            color='red'
        )
        axes[1].set_xlabel('Confidence', fontsize=12)
        axes[1].set_ylabel('Uncertainty (Entropy)', fontsize=12)
        axes[1].set_title('Uncertainty vs Confidence', fontsize=13)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Uncertainty analysis saved: {save_path}")
        
        plt.close()
    
    def plot_all(self, output_dir='./results'):
        """
        Generate all evaluation plots.
        
        Args:
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plot_roc_curve(output_dir / 'roc_curve.png')
        self.plot_confusion_matrix(output_dir / 'confusion_matrix.png')
        self.plot_uncertainty_analysis(output_dir / 'uncertainty_analysis.png')
        
        logging.info(f"All plots saved to {output_dir}")


def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['val_acc'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].grid(alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(epochs, history['lr'], 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(alpha=0.3)
    
    # Loss comparison
    axes[1, 1].plot(epochs, history['train_loss'], 'b-', alpha=0.3, label='Train')
    axes[1, 1].plot(epochs, history['val_loss'], 'r-', alpha=0.3, label='Val')
    axes[1, 1].fill_between(epochs, history['train_loss'], alpha=0.2, color='blue')
    axes[1, 1].fill_between(epochs, history['val_loss'], alpha=0.2, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Loss Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Training history plot saved: {save_path}")
    
    plt.close()


if __name__ == '__main__':
    # Test evaluator
    print("Testing evaluator...")
    
    # Create dummy predictions
    n_samples = 1000
    probabilities = np.random.rand(n_samples)
    labels_true = np.random.randint(0, 2, n_samples)
    
    # Compute metrics manually
    predictions = (probabilities > 0.5).astype(int)
    acc = accuracy_score(labels_true, predictions)
    auc = roc_auc_score(labels_true, probabilities)
    
    print(f"Dummy accuracy: {acc:.3f}")
    print(f"Dummy AUC: {auc:.3f}")
    
    print("\nTest passed!")
