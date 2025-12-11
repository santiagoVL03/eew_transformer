"""
Comprehensive benchmarking and comparison system for EEW models.

Features:
- Evaluation of EEW_Transformer model
- Training and evaluation of baseline models
- Performance metrics (accuracy, F1, AUC, etc.)
- Inference speed and throughput measurement
- Model size analysis
- Visualization generation
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from eew.baselines import (
    MLPBaseline, CNN1DBaseline, RandomForestBaseline, SVMBaseline,
    FeatureExtractor
)


class ModelBenchmark:
    """
    Comprehensive benchmarking for earthquake detection models.
    """
    
    def __init__(self, device='cpu', verbose=True):
        """
        Args:
            device: 'cpu' or 'cuda'
            verbose: Print logging messages
        """
        self.device = device
        self.verbose = verbose
        
        if verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
    
    def log(self, msg):
        """Log message if verbose."""
        if self.verbose:
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
    
    def evaluate_model(
        self,
        model,
        test_loader: DataLoader,
        model_name: str = 'Model',
        is_neural_net: bool = True,
        use_mc_dropout: bool = False,
        threshold: float = 0.5
    ) -> Dict:
        """
        Evaluate model on test set using Evaluator for neural nets.
        
        Args:
            model: Model to evaluate
            test_loader: DataLoader for test set
            model_name: Name of model for logging
            is_neural_net: If True, expects PyTorch model; else sklearn-like
            use_mc_dropout: Use MC Dropout for uncertainty (only for neural nets)
            threshold: Classification threshold (default 0.5)
        
        Returns:
            Dictionary of metrics, probs, preds, and labels
        """
        self.log(f"Evaluating {model_name}...")
        
        if is_neural_net:
            # Use Evaluator class for neural networks
            from eew.evaluator import Evaluator
            
            evaluator = Evaluator(
                model=model,
                test_loader=test_loader,
                device=self.device,
                threshold=threshold
            )
            
            # Run evaluation
            metrics = evaluator.evaluate(
                use_mc_dropout=use_mc_dropout,
                n_passes=10 if use_mc_dropout else 1
            )
            
            # Add model name for logging
            metrics['model_name'] = model_name
            
            self.log(f"✓ {model_name} evaluated: "
                    f"Acc={metrics.get('accuracy', 0):.3f}, "
                    f"F1={metrics.get('f1', 0):.3f}")
            
            return {
                'metrics': metrics,
                'probs': evaluator.probabilities,
                'preds': evaluator.predictions,
                'labels': evaluator.labels_true
            }
        else:
            # sklearn-like model evaluation
            all_probs = []
            all_labels = []
            
            start_time = time.time()
            n_samples = 0
            
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    waveforms, labels = batch
                else:
                    waveforms = batch
                    labels = None
                
                if isinstance(waveforms, torch.Tensor):
                    waveforms = waveforms.cpu().numpy()
                
                # Predict using waveforms - model will extract features internally
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(waveforms)
                    if probs.ndim > 1:
                        probs = probs[:, 1]  # Take positive class probability
                else:
                    preds = model.predict(waveforms)
                    probs = preds.astype(float)
                
                all_probs.extend(probs)
                
                if labels is not None:
                    all_labels.extend(labels.numpy().flatten() if isinstance(labels, torch.Tensor) else labels.flatten())
                
                n_samples += waveforms.shape[0]
            
            inference_time = time.time() - start_time
            
            # Convert to arrays
            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels) if all_labels else None
            all_preds = (all_probs > threshold).astype(int)
            
            # Compute metrics
            metrics = {
                'model_name': model_name,
                'n_samples': n_samples,
                'inference_time_total': inference_time,
                'inference_time_per_sample': inference_time / n_samples if n_samples > 0 else 0,
                'throughput': n_samples / inference_time if inference_time > 0 else 0,
            }
            
            # Add predictive metrics if labels available
            if all_labels is not None and len(all_labels) > 0:
                metrics.update({
                    'accuracy': accuracy_score(all_labels, all_preds),
                    'precision': precision_score(all_labels, all_preds, zero_division=0),
                    'recall': recall_score(all_labels, all_preds, zero_division=0),
                    'f1': f1_score(all_labels, all_preds, zero_division=0),
                    'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0,
                })
            
            self.log(f"✓ {model_name} evaluated: "
                    f"Acc={metrics.get('accuracy', 0):.3f}, "
                    f"F1={metrics.get('f1', 0):.3f}, "
                    f"Throughput={metrics.get('throughput', 0):.1f} samples/s")
            
            return {
                'metrics': metrics,
                'probs': all_probs,
                'preds': all_preds,
                'labels': all_labels
            }
    
    def get_model_size(self, model, is_neural_net: bool = True) -> Dict:
        """
        Compute model size statistics.
        
        Args:
            model: Model
            is_neural_net: If True, expects PyTorch model
        
        Returns:
            Dictionary with size info (parameters, MB)
        """
        if is_neural_net:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate size in MB (float32 = 4 bytes per parameter)
            size_mb = (total_params * 4) / (1024 * 1024)
            
            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'size_mb': size_mb
            }
        else:
            # sklearn model size estimation
            import pickle
            try:
                size_bytes = len(pickle.dumps(model))
                size_mb = size_bytes / (1024 * 1024)
                return {
                    'serialized_size_bytes': size_bytes,
                    'size_mb': size_mb
                }
            except:
                return {'size_mb': 'unknown'}
    
    def train_baseline(
        self,
        baseline_type: str,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 20,
        learning_rate: float = 0.001
    ):
        """
        Train a baseline model.
        
        Args:
            baseline_type: 'mlp', 'cnn1d', 'random_forest', 'svm'
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
            epochs: Number of epochs (for neural nets)
            learning_rate: Learning rate (for neural nets)
        
        Returns:
            Trained model
        """
        self.log(f"Training {baseline_type} baseline...")
        
        if baseline_type == 'mlp':
            model = MLPBaseline().to(self.device)
            return self._train_neural_net(
                model, train_loader, val_loader, epochs, learning_rate
            )
        elif baseline_type == 'cnn1d':
            model = CNN1DBaseline().to(self.device)
            return self._train_neural_net(
                model, train_loader, val_loader, epochs, learning_rate
            )
        elif baseline_type == 'random_forest':
            model = RandomForestBaseline()
            return self._train_sklearn_model(model, train_loader)
        elif baseline_type == 'svm':
            model = SVMBaseline()
            return self._train_sklearn_model(model, train_loader)
        else:
            raise ValueError(f"Unknown baseline: {baseline_type}")
    
    def _train_neural_net(
        self,
        model,
        train_loader,
        val_loader,
        epochs,
        learning_rate
    ):
        """Train neural network baseline."""
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    waveforms, labels = batch
                else:
                    waveforms = batch
                    labels = None
                
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device).float()
                
                logits = model(waveforms)
                loss = criterion(logits, labels.unsqueeze(1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                self.log(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        return model
    
    def _train_sklearn_model(self, model, train_loader):
        """Train sklearn-like baseline."""
        all_waveforms = []
        all_labels = []
        
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                waveforms, labels = batch
            else:
                waveforms = batch
                labels = None
            
            if isinstance(waveforms, torch.Tensor):
                waveforms = waveforms.cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            
            all_waveforms.append(waveforms)
            if labels is not None:
                all_labels.append(labels)
        
        all_waveforms = np.concatenate(all_waveforms, axis=0)
        all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
        
        # Pass waveforms directly - models will extract features internally
        if all_labels is not None:
            model.fit(all_waveforms, all_labels)
        else:
            model.fit(all_waveforms, np.zeros(len(all_waveforms)))
        
        return model
    
    @staticmethod
    def compute_roc_curve(labels, probs):
        """Compute ROC curve."""
        if len(np.unique(labels)) < 2:
            return None, None, None
        
        fpr, tpr, thresholds = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        
        return fpr, tpr, auc
    
    @staticmethod
    def compute_confusion_matrix(labels, preds):
        """Compute confusion matrix."""
        return confusion_matrix(labels, preds)


class ComparisonVisualizer:
    """Generate visualizations for model comparison."""
    
    @staticmethod
    def plot_confusion_matrices(results_dict, save_path=None):
        """
        Plot confusion matrices for all models.
        
        Args:
            results_dict: Dictionary of {model_name: result_dict}
            save_path: Path to save figure
        """
        n_models = len(results_dict)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (model_name, result) in enumerate(results_dict.items()):
            if idx >= 4:
                break
            
            labels = result['labels']
            preds = result['preds']
            
            if labels is None or len(labels) == 0:
                continue
            
            cm = confusion_matrix(labels, preds)
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                ax=axes[idx], cbar=False
            )
            axes[idx].set_title(f'{model_name}\nConfusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for idx in range(len(results_dict), 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_roc_curves(results_dict, save_path=None):
        """
        Plot ROC curves for all models.
        
        Args:
            results_dict: Dictionary of {model_name: result_dict}
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, result in results_dict.items():
            labels = result['labels']
            probs = result['probs']
            
            if labels is None or len(labels) == 0:
                continue
            
            fpr, tpr, auc = ModelBenchmark.compute_roc_curve(labels, probs)
            
            if fpr is not None and tpr is not None:
                ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - 2-Second P-Wave Detection')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_metrics_comparison(results_dict, metrics=['accuracy', 'f1', 'precision', 'recall'], save_path=None):
        """
        Plot bar chart comparing metrics across models.
        
        Args:
            results_dict: Dictionary of {model_name: metrics_dict}
            metrics: List of metric names to plot
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
        
        if len(metrics) == 1:
            axes = [axes]
        
        model_names = list(results_dict.keys())
        
        for idx, metric in enumerate(metrics):
            values = [results_dict[m]['metrics'].get(metric, 0) for m in model_names]
            
            colors = ['#1f77b4' if 'EEW_Transformer' not in m else '#ff7f0e' for m in model_names]
            
            axes[idx].bar(range(len(model_names)), values, color=colors)
            axes[idx].set_title(f'{metric.capitalize()}')
            axes[idx].set_ylabel('Score')
            axes[idx].set_xticks(range(len(model_names)))
            axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_speed_comparison(results_dict, save_path=None):
        """
        Plot inference speed comparison.
        
        Args:
            results_dict: Dictionary of {model_name: metrics_dict}
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        model_names = list(results_dict.keys())
        throughput = [results_dict[m]['metrics']['throughput'] for m in model_names]
        latency = [results_dict[m]['metrics']['inference_time_per_sample'] * 1000 for m in model_names]  # ms
        
        colors = ['#1f77b4' if 'EEW_Transformer' not in m else '#ff7f0e' for m in model_names]
        
        # Throughput
        axes[0].bar(range(len(model_names)), throughput, color=colors)
        axes[0].set_title('Throughput (Samples/Second)')
        axes[0].set_ylabel('Samples/s')
        axes[0].set_xticks(range(len(model_names)))
        axes[0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Latency
        axes[1].bar(range(len(model_names)), latency, color=colors)
        axes[1].set_title('Latency per Sample')
        axes[1].set_ylabel('Milliseconds')
        axes[1].set_xticks(range(len(model_names)))
        axes[1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_model_size_comparison(size_dict, save_path=None):
        """
        Plot model size comparison.
        
        Args:
            size_dict: Dictionary of {model_name: size_info_dict}
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names = list(size_dict.keys())
        sizes = []
        
        for m in model_names:
            size_mb = size_dict[m].get('size_mb', 0)
            if isinstance(size_mb, str):
                size_mb = 0
            sizes.append(size_mb)
        
        colors = ['#1f77b4' if 'EEW_Transformer' not in m else '#ff7f0e' for m in model_names]
        
        bars = ax.bar(range(len(model_names)), sizes, color=colors)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f} MB',
                   ha='center', va='bottom')
        
        ax.set_title('Model Size Comparison')
        ax.set_ylabel('Size (MB)')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def generate_summary_report(results_dict, size_dict, save_path=None):
        """
        Generate a comprehensive text summary report.
        
        Args:
            results_dict: Dictionary of evaluation results
            size_dict: Dictionary of model sizes
            save_path: Path to save report
        
        Returns:
            Report string
        """
        report = "=" * 80 + "\n"
        report += "EEW TRANSFORMER - COMPREHENSIVE EVALUATION REPORT\n"
        report += "2-Second P-Wave Detection Benchmark\n"
        report += "=" * 80 + "\n\n"
        
        # Summary table
        report += "MODEL COMPARISON SUMMARY\n"
        report += "-" * 120 + "\n"
        report += f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<10} {'Throughput':<15}\n"
        report += "-" * 120 + "\n"
        
        for model_name, result in results_dict.items():
            metrics = result['metrics']
            acc = metrics.get('accuracy', 0)
            prec = metrics.get('precision', 0)
            rec = metrics.get('recall', 0)
            f1 = metrics.get('f1', 0)
            auc = metrics.get('auc', 0)
            throughput = metrics.get('throughput', 0)
            
            report += f"{model_name:<20} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {auc:<10.4f} {throughput:<15.1f}\n"
        
        report += "-" * 120 + "\n\n"
        
        # Model sizes
        report += "MODEL SIZE & PARAMETERS\n"
        report += "-" * 60 + "\n"
        report += f"{'Model':<20} {'Size (MB)':<15} {'Parameters':<20}\n"
        report += "-" * 60 + "\n"
        
        for model_name, size_info in size_dict.items():
            size_mb = size_info.get('size_mb', 'N/A')
            params = size_info.get('total_parameters', 'N/A')
            
            if isinstance(size_mb, str):
                report += f"{model_name:<20} {size_mb:<15} {params:<20}\n"
            else:
                report += f"{model_name:<20} {size_mb:<15.4f} {params:<20}\n"
        
        report += "\n"
        
        # Key insights
        report += "KEY INSIGHTS\n"
        report += "-" * 80 + "\n"
        
        # Find best model for each metric
        metrics_names = ['accuracy', 'f1', 'auc', 'throughput']
        for metric in metrics_names:
            best_model = max(
                results_dict.items(),
                key=lambda x: x[1]['metrics'].get(metric, 0)
            )
            best_value = best_model[1]['metrics'].get(metric, 0)
            report += f"• Best {metric.upper()}: {best_model[0]} ({best_value:.4f})\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
