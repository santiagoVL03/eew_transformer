"""
Lightweight evaluation runner.

Loads a trained checkpoint and runs the existing Evaluator to produce
`test_metrics.json` and the plots (ROC, confusion matrix, uncertainty) under
the experiment `save_dir` (e.g. `./results_phase2`).

Usage:
    python evaluate.py --save_dir ./results_phase2 --checkpoint ./results_phase2/checkpoints/best_model.pth
"""
import argparse
import json
import logging
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Import compatibility module BEFORE other imports to register TFEQ class
import utils  # This allows torch.load to find utils.TFEQ.TFEQ

# Add compatibility for old PyTorch versions that used _LinearWithBias
# In newer PyTorch versions, this was merged into nn.Linear
if not hasattr(nn.modules.linear, '_LinearWithBias'):
    nn.modules.linear._LinearWithBias = nn.Linear

from eew.model import create_Transformer_model
from eew.data_loader import STEADLoader
from eew.evaluator import Evaluator
from eew.utils import setup_logging


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--save_dir', type=str, required=True, help='Experiment folder (where checkpoint and outputs live)')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file (overrides save_dir/checkpoints/best_model.pth)')
    p.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    p.add_argument('--batch', type=int, default=16, help='Batch size for evaluation')
    p.add_argument('--window', type=float, default=2.0)
    p.add_argument('--sampling_rate', type=int, default=100)
    p.add_argument('--d_model', type=int, default=96)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--num_layers', type=int, default=6)
    p.add_argument('--dim_feedforward', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--classifier_hidden', type=int, default=200)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--mc_dropout', action='store_true')
    p.add_argument('--n_mc_passes', type=int, default=10)
    p.add_argument('--region', type=str, default='latam')
    p.add_argument('--train_subset', type=int, default=120000, help='Limit dataset to N samples (same as training)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup simple logging
    setup_logging(log_file=str(save_dir / 'eval.log'), level=logging.INFO)

    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'

    logging.info(f'Running evaluation in {save_dir} on device {device}')

    # Build loader (use lazy loading to avoid large memory) -------------------------------------------------
    loader = STEADLoader(data_path=None, region=args.region, cache_dir=save_dir / 'data' / 'stead')
    loader.load_dataset()
    loader.filter_by_region()

    # ========================================================================
    # CRITICAL RAM OPTIMIZATION: Limit dataset size (same as training)
    # ========================================================================
    total_samples = len(loader.filtered_indices) if loader.filtered_indices else 0
    
    if args.train_subset and args.train_subset < total_samples and loader.filtered_indices:
        logging.info(f"\n*** LIMITING DATASET SIZE FOR RAM OPTIMIZATION ***")
        logging.info(f"Original samples: {total_samples:,}")
        logging.info(f"Using subset: {args.train_subset:,}")
        logging.info(f"This ensures same data split as training\n")
        
        # Use same seed and sampling as training to get EXACT same subset
        np.random.seed(args.seed)
        subset_indices = np.random.choice(
            np.array(loader.filtered_indices),
            size=args.train_subset,
            replace=False
        ).tolist()
        loader.filtered_indices = subset_indices

    # create dataloaders with lazy_load=True to keep memory use low
    train_loader, val_loader, test_loader = loader.create_dataloaders(
        waveforms=None,
        labels=None,
        batch_size=args.batch,
        train_ratio=0.7,
        val_ratio=0.15,
        shuffle=False,
        num_workers=0,
        transform_train=None,
        transform_val=None,
        lazy_load=True,
        window_size=args.window,
        sampling_rate=args.sampling_rate
    )
    
    logging.info(f"Test set size: {len(test_loader.dataset)} samples")

    # Create model (match training config: mc_dropout True during creation)
    seq_len = int(args.window * args.sampling_rate)
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

    # Load checkpoint
    ckpt_path = Path(args.checkpoint) if args.checkpoint else save_dir / 'checkpoints' / 'best_model.pth'
    if not ckpt_path.exists():
        logging.error(f'Checkpoint not found: {ckpt_path}')
        return

    logging.info(f'Loading checkpoint: {ckpt_path}')
    # Use weights_only=False to load checkpoint with custom classes
    # This is safe since we trust our own checkpoint file
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Check if checkpoint is a dictionary or the model directly
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Standard checkpoint format with metadata
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    elif isinstance(checkpoint, dict):
        # Dictionary but no 'model_state_dict' key - try loading directly
        model.load_state_dict(checkpoint)
        logging.info("Loaded model state dict from checkpoint")
    else:
        # Checkpoint is the model object itself - extract state dict
        logging.info("Checkpoint contains model object directly")
        try:
            model.load_state_dict(checkpoint.state_dict())
            logging.info("Loaded state dict from model object")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            logging.info("Attempting to use the loaded model directly...")
            model = checkpoint
            model = model.to(device)

    # Create evaluator and run evaluation
    evaluator = Evaluator(model=model, test_loader=test_loader, device=device, threshold=args.threshold)

    try:
        metrics = evaluator.evaluate(use_mc_dropout=args.mc_dropout, n_passes=args.n_mc_passes)
    except Exception as e:
        logging.exception('Evaluation failed')
        raise

    # Save metrics
    metrics_path = save_dir / 'test_metrics.json'
    metrics_serializable = {k: float(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else None for k, v in metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    logging.info(f'Metrics saved: {metrics_path}')

    # Generate and save plots
    evaluator.plot_all(output_dir=str(save_dir / 'plots'))

    logging.info('Evaluation completed successfully')


if __name__ == '__main__':
    main()
