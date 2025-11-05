"""
Main experiment script for Transformer earthquake detection.

Usage:
    python run_experiment.py --region chile --window 2 --batch 64 --epochs 50 --use_gpu
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch
import torch.optim as optim

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from eew.model import create_Transformer_model
from eew.data_loader import STEADLoader
from eew.preprocessing import preprocess_waveform
from eew.augmentation import WaveformAugmenter, NoAugmentation
from eew.trainer import Trainer, OnlineTrainer
from eew.evaluator import Evaluator, plot_training_history
from eew.utils import (
    set_seed, get_device, count_parameters, setup_logging
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Transformer Earthquake Detection Experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='stead',
                       help='Dataset to use')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to local STEAD files (optional)')
    parser.add_argument('--region', type=str, default='latam',
                       choices=['chile', 'latam'],
                       help='Region filter: chile or latam')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to load (for testing)')
    
    # Preprocessing arguments
    parser.add_argument('--window', type=float, default=2.0,
                       choices=[2.0, 4.0, 8.0],
                       help='Window size in seconds')
    parser.add_argument('--phase', type=str, default='both',
                       choices=['P', 'S', 'both'],
                       help='Phase to use: P, S, or both')
    parser.add_argument('--sampling_rate', type=int, default=100,
                       help='Sampling rate in Hz')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=64,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of encoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=256,
                       help='FFN hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--classifier_hidden', type=int, default=200,
                       help='Classifier hidden size')
    
    # Training arguments
    parser.add_argument('--mode', type=str, default='offline',
                       choices=['offline', 'online'],
                       help='Training mode: offline or online (streaming)')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--early_stopping', type=int, default=10,
                       help='Early stopping patience')
    
    # Memory management
    parser.add_argument('--lazy_load', action='store_true',
                       help='Load data on-the-fly (memory-efficient, recommended for 16GB RAM)')
    parser.add_argument('--preload_samples', type=int, default=None,
                       help='Number of samples to preload (None for lazy loading mode)')
    parser.add_argument('--train_subset', type=int, default=50000,
                       help='Maximum training samples to use (reduces RAM and training time)')

    
    # Augmentation
    parser.add_argument('--augment', action='store_true',
                       help='Enable data augmentation')
    parser.add_argument('--noise_snr_min', type=float, default=5.0,
                       help='Minimum SNR for noise augmentation (dB)')
    parser.add_argument('--noise_snr_max', type=float, default=20.0,
                       help='Maximum SNR for noise augmentation (dB)')
    
    # Evaluation
    parser.add_argument('--mc_dropout', action='store_true',
                       help='Use MC Dropout for uncertainty estimation')
    parser.add_argument('--n_mc_passes', type=int, default=10,
                       help='Number of MC dropout passes')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    
    # System
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU if available')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--deterministic', action='store_true',
                       help='Enable deterministic mode (reproducible but slower, NOT recommended with augmentation)')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--checkpoint', type=str, default='./results/checkpoint.pth',
                       help='Path to checkpoint to resume training')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Log file path')
    
    return parser.parse_args()


def main():
    """Main experiment function."""
    
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = args.log_file or str(save_dir / 'experiment.log')
    setup_logging(log_file=log_file, level=logging.INFO)
    
    # Log experiment configuration
    logging.info("=" * 70)
    logging.info("Transformer Earthquake Detection Experiment")
    logging.info("=" * 70)
    logging.info(f"Configuration:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")
    logging.info("=" * 70 + "\n")
    
    # Set random seed
    # IMPORTANT: Do NOT use deterministic=True with augmentation!
    # It causes augmentation to produce the same outputs every epoch.
    set_seed(args.seed, deterministic=args.deterministic)
    
    if args.augment and args.deterministic:
        logging.warning("=" * 70)
        logging.warning("WARNING: deterministic=True with augmentation enabled!")
        logging.warning("This will cause augmentation to produce identical outputs every epoch.")
        logging.warning("Recommend: Run without --deterministic flag")
        logging.warning("=" * 70 + "\n")
    
    # Get device
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    
    # Calculate sequence length
    seq_len = int(args.window * args.sampling_rate)
    logging.info(f"Sequence length: {seq_len} samples ({args.window}s @ {args.sampling_rate}Hz)\n")
    
    # ========================================================================
    # Load and preprocess data
    # ========================================================================
    
    logging.info("Loading STEAD dataset...")
    loader = STEADLoader(
        data_path=args.data_path,
        region=args.region,
        cache_dir=save_dir / 'data' / 'stead'
    )
    
    # Load dataset
    loader.load_dataset()
    
    # Filter by region
    loader.filter_by_region()
    
    # ========================================================================
    # CRITICAL RAM OPTIMIZATION: Limit dataset size
    # ========================================================================
    
    # Limit total samples to prevent RAM issues
    total_samples = len(loader.filtered_indices) # type: ignore
    
    if args.train_subset and args.train_subset < total_samples:
        logging.info(f"\n*** LIMITING DATASET SIZE FOR RAM OPTIMIZATION ***")
        logging.info(f"Original samples: {total_samples:,}")
        logging.info(f"Using subset: {args.train_subset:,}")
        logging.info(f"This reduces RAM usage and training time significantly\n")
        
        # Randomly sample indices
        np.random.seed(args.seed)
        subset_indices = np.random.choice(
            loader.filtered_indices, # type: ignore
            size=args.train_subset,
            replace=False
        ).tolist() # type: ignore
        loader.filtered_indices = subset_indices
    
    # Determine if we should use lazy loading
    use_lazy_load = args.lazy_load or (args.preload_samples is None and args.max_samples is None)
    
    if use_lazy_load:
        logging.info("\n*** Using LAZY LOADING mode (memory-efficient) ***")
        logging.info("Data will be loaded on-the-fly during training")
        logging.info("This is recommended for systems with 16GB RAM or less\n")
        
        # Don't preload waveforms, just get metadata for statistics
        # We'll use a small sample just to show statistics
        sample_size = min(1000, len(loader.filtered_indices)) # type: ignore
        waveforms_sample, labels_sample, _ = loader.load_waveforms(
            phase=args.phase,
            window_size=args.window,
            sampling_rate=args.sampling_rate,
            max_samples=sample_size
        )
        
        logging.info(f"\nDataset statistics (sampled from {sample_size} waveforms):")
        logging.info(f"  Total filtered samples: {len(loader.filtered_indices)}") # type: ignore
        logging.info(f"  Earthquakes (sample): {np.sum(labels_sample)} ({np.sum(labels_sample)/len(labels_sample)*100:.1f}%)")
        logging.info(f"  Noise (sample): {len(labels_sample) - np.sum(labels_sample)} ({(1-np.sum(labels_sample)/len(labels_sample))*100:.1f}%)")
        logging.info(f"  Waveform shape: {waveforms_sample.shape}\n")
        
        # Estimate class weights from sample
        n_earthquakes = np.sum(labels_sample)
        n_noise = len(labels_sample) - n_earthquakes
        
        # Set waveforms and labels to None for lazy loading
        waveforms = None
        labels = None
        
    else:
        logging.info("\n*** Using PRELOADED mode (loads all data into memory) ***")
        logging.info("Warning: This may use significant RAM for large datasets\n")
        
        # Load waveforms
        max_samples_to_load = args.preload_samples or args.max_samples
        waveforms, labels, metadata = loader.load_waveforms(
            phase=args.phase,
            window_size=args.window,
            sampling_rate=args.sampling_rate,
            max_samples=max_samples_to_load
        )
        
        logging.info(f"\nDataset statistics:")
        logging.info(f"  Total samples: {len(waveforms)}")
        logging.info(f"  Earthquakes: {np.sum(labels)} ({np.sum(labels)/len(labels)*100:.1f}%)")
        logging.info(f"  Noise: {len(labels) - np.sum(labels)} ({(1-np.sum(labels)/len(labels))*100:.1f}%)")
        logging.info(f"  Waveform shape: {waveforms.shape}\n")
        
        # Compute class weights
        n_earthquakes = np.sum(labels)
        n_noise = len(labels) - n_earthquakes
    
    # Compute class weights for imbalanced data
    if n_earthquakes > 0 and n_noise > 0:
        # Weight for positive class (earthquake)
        pos_weight =  n_noise / n_earthquakes
        logging.info(f"Class imbalance - pos_weight: {pos_weight:.2f}\n")
    else:
        pos_weight = 1.0
        logging.warning("One class is missing! Using pos_weight=1.0\n")
    
    # ========================================================================
    # Create data loaders
    # ========================================================================
    
    # Setup augmentation
    if args.augment:
        transform_train = WaveformAugmenter(
            add_noise=True,
            noise_snr_range=(args.noise_snr_min, args.noise_snr_max),
            scale_amplitude=True,
            time_shift=True,
            sampling_rate=args.sampling_rate,
            p=0.5
        )
        logging.info("Data augmentation enabled for training set")
    else:
        transform_train = None
        logging.info("No data augmentation")
    
    transform_val = None  # No augmentation for validation/test
    
    # CRITICAL: Reduce num_workers for lazy loading to prevent RAM explosion
    # Each worker loads data independently, multiplying RAM usage
    effective_num_workers = 0 if use_lazy_load else min(args.num_workers, 2)
    
    if use_lazy_load and args.num_workers > 0:
        logging.info(f"\n*** RAM OPTIMIZATION: Setting num_workers=0 for lazy loading ***")
        logging.info(f"Requested: {args.num_workers} workers")
        logging.info(f"Using: {effective_num_workers} workers (prevents RAM explosion)")
        logging.info(f"With lazy loading, multiple workers multiply RAM usage!\n")
    
    train_loader, val_loader, test_loader = loader.create_dataloaders(
        waveforms=waveforms,
        labels=labels,
        batch_size=args.batch,
        train_ratio=0.7,
        val_ratio=0.15,
        shuffle=True,
        num_workers=effective_num_workers,
        transform_train=transform_train,
        transform_val=transform_val,
        lazy_load=use_lazy_load,
        window_size=args.window,
        sampling_rate=args.sampling_rate
    )
    
    # ========================================================================
    # Create model
    # ========================================================================
    
    logging.info("\nCreating Transformer model...")
    model = create_Transformer_model(
        seq_len=seq_len,
        input_channels=3,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        classifier_hidden=args.classifier_hidden,
        mc_dropout=True
    )
    
    n_params = count_parameters(model)
    logging.info(f"Model created with {n_params:,} parameters")
    logging.info(f"Target: ~235,000 parameters (current: {n_params:,})\n")
    
    # ========================================================================
    # Setup optimizer and scheduler
    # ========================================================================
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        logging.info("Using ReduceLROnPlateau scheduler")
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
        logging.info("Using CosineAnnealingLR scheduler")
    else:
        scheduler = None
        logging.info("No learning rate scheduler")
    
    # ========================================================================
    # Training
    # ========================================================================
    
    if args.mode == 'offline':
        logging.info("Starting offline training...\n")
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_amp=True,
            class_weights=pos_weight,
            early_stopping_patience=args.early_stopping,
            checkpoint_dir=save_dir / 'checkpoints'
        )
        
        # Train
        history = trainer.train(num_epochs=args.epochs)
        
        # Save training history
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            history_serializable = {
                k: [float(v) for v in vals] for k, vals in history.items()
            }
            json.dump(history_serializable, f, indent=2)
        logging.info(f"Training history saved: {history_path}")
        
        # Plot training curves
        plot_training_history(history, save_path=save_dir / 'training_curves.png')
        
        # Load best model for evaluation
        best_checkpoint = save_dir / 'checkpoints' / 'best_model.pth'
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    elif args.mode == 'online':
        logging.info("Starting online (streaming) training...\n")
        
        # Combine all data for online mode
        from torch.utils.data import ConcatDataset
        full_dataset = ConcatDataset([
            train_loader.dataset,
            val_loader.dataset,
            test_loader.dataset
        ])
        from torch.utils.data import DataLoader
        full_loader = DataLoader(
            full_dataset,
            batch_size=1,  # Process one sample at a time
            shuffle=False
        )
        
        online_trainer = OnlineTrainer(
            model=model,
            data_loader=full_loader,
            optimizer=optimizer,
            device=device,
            batch_size=1,
            update_frequency=1
        )
        
        predictions, labels_true, timestamps = online_trainer.run()
        
        # Save online results
        online_results = {
            'predictions': predictions.tolist(),
            'labels': labels_true.tolist(),
            'timestamps': timestamps.tolist()
        }
        with open(save_dir / 'online_results.json', 'w') as f:
            json.dump(online_results, f)
        
        logging.info("Online mode completed. Skipping offline evaluation.\n")
        return
    
    # ========================================================================
    # Evaluation
    # ========================================================================
    
    logging.info("\n" + "=" * 70)
    logging.info("Evaluation on Test Set")
    logging.info("=" * 70 + "\n")
    
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        threshold=args.threshold
    )
    
    # Evaluate
    metrics = evaluator.evaluate(
        use_mc_dropout=args.mc_dropout,
        n_passes=args.n_mc_passes
    )
    
    # Save metrics
    metrics_path = save_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        # Convert numpy types to Python types
        metrics_serializable = {k: float(v) if not np.isnan(v) else None 
                               for k, v in metrics.items()}
        json.dump(metrics_serializable, f, indent=2)
    logging.info(f"Metrics saved: {metrics_path}")
    
    # Generate plots
    logging.info("\nGenerating evaluation plots...")
    evaluator.plot_all(output_dir=save_dir / 'plots')
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    logging.info("\n" + "=" * 70)
    logging.info("Experiment Summary")
    logging.info("=" * 70)
    logging.info(f"Region: {args.region}")
    logging.info(f"Window size: {args.window}s ({seq_len} samples)")
    logging.info(f"Model parameters: {n_params:,}")
    logging.info(f"Training samples: {len(train_loader.dataset)}")
    logging.info(f"Test samples: {len(test_loader.dataset)}")
    logging.info(f"")
    logging.info(f"Test Accuracy:  {metrics['accuracy']:.4f}")
    logging.info(f"Test Precision: {metrics['precision']:.4f}")
    logging.info(f"Test Recall:    {metrics['recall']:.4f}")
    logging.info(f"Test F1:        {metrics['f1']:.4f}")
    logging.info(f"Test ROC-AUC:   {metrics['roc_auc']:.4f}")
    if args.mc_dropout:
        logging.info(f"Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")
    logging.info(f"")
    logging.info(f"Results saved to: {save_dir}")
    logging.info("=" * 70 + "\n")
    
    logging.info("Experiment completed successfully!")


if __name__ == '__main__':
    main()
