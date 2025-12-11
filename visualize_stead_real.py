#!/usr/bin/env python3
"""
Advanced visualization using real STEAD dataset.

This script loads real seismic data from STEAD and performs comprehensive analysis:
- Load real earthquake and noise samples
- Simulate real-time detection on long traces
- Generate professional figures for papers/theses
- Compare model predictions with true labels

Usage:
    python visualize_stead_real.py --checkpoint path/to/checkpoint.pth \
                                    --output-dir figures/stead_analysis/
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from eew.model import create_Transformer_model
from eew.visualization import (
    plot_full_trace_with_P,
    plot_window_prediction,
    plot_grid_of_samples,
    plot_realtime_detection,
    plot_detection_comparison,
    simulate_realtime_detection,
    load_stead_sample,
    load_random_samples,
    extract_window,
    get_detection_statistics,
)


def load_model(checkpoint_path: str, device: str = 'cuda') -> torch.nn.Module:
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Detect model architecture from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Get d_model from state dict
    embedding_weight = state_dict.get('embedding.weight')
    if embedding_weight is not None:
        d_model = embedding_weight.shape[0]
    else:
        d_model = 64  # Default
    
    # Get dim_feedforward from FFN weights
    ffn_weight = state_dict.get('encoder_layers.0.ffn.0.weight')
    if ffn_weight is not None:
        dim_feedforward = ffn_weight.shape[0]
    else:
        dim_feedforward = 4 * d_model  # Default: 4x d_model
    
    logger.info(f"Detected architecture: d_model={d_model}, dim_feedforward={dim_feedforward}")
    
    model = create_Transformer_model(
        seq_len=200,
        input_channels=3,
        d_model=d_model,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=dim_feedforward,
        dropout=0.1,
        classifier_hidden=200
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully!")
    return model


def analyze_single_event(
    model,
    sample,
    device: str,
    output_dir: Path,
    sample_name: str = "event_001"
):
    """
    Analyze a single event: full trace + real-time simulation.
    
    Args:
        model: Trained model
        sample: Sample dict from load_stead_sample
        device: Device to run on
        output_dir: Output directory
        sample_name: Name for this sample
    """
    logger.info(f"\nAnalyzing sample: {sample_name}")
    
    waveform = sample['waveform']
    true_label = sample['label']
    p_arrival = sample.get('p_arrival_sample')
    metadata = sample.get('metadata', {})
    
    # Ensure correct shape
    if waveform.shape[0] != 3:
        waveform = waveform.T
    
    logger.info(f"  Shape: {waveform.shape}, Label: {'Event' if true_label else 'Noise'}")
    
    # 1. Plot full trace
    fig, axes = plot_full_trace_with_P(
        waveform,
        sampling_rate=100.0,
        p_arrival_sample=p_arrival,
        event_metadata={
            'magnitude': metadata.get('trace_magnitude', 'N/A'),
            'event_id': metadata.get('event_id', sample_name),
            'station': metadata.get('station_network_code', 'N/A'),
        },
        title=f"STEAD Event Analysis: {sample_name}",
        save_path=output_dir / f"{sample_name}_full_trace.png"
    )
    logger.info(f"  ✓ Saved full trace")
    
    # 2. Simulate real-time detection if waveform is long enough
    if waveform.shape[1] >= 400:  # At least 4 seconds
        results = simulate_realtime_detection(
            model,
            waveform,
            sampling_rate=100.0,
            window_size=2.0,
            true_p_arrival_sample=p_arrival,
            device=device
        )
        
        # Plot real-time detection
        fig = plot_realtime_detection(
            waveform,
            results['window_predictions'],
            sampling_rate=100.0,
            p_arrival_sample=p_arrival,
            title=f"Real-Time Detection: {sample_name}",
            save_path=output_dir / f"{sample_name}_realtime.png"
        )
        
        # Print statistics
        stats = get_detection_statistics(results)
        logger.info(f"  Windows processed: {stats['total_windows']}")
        logger.info(f"  Mean probability: {stats['mean_probability']:.3f}")
        logger.info(f"  Detection rate: {stats['detection_rate']:.1%}")
        
        if 'detection_delay' in stats and p_arrival is not None:
            logger.info(f"  Detection delay: {stats['detection_delay']:+.2f}s")
        
        logger.info(f"  ✓ Saved real-time detection")


def analyze_random_batch(
    model,
    stead_dataset,
    device: str,
    output_dir: Path,
    n_samples: int = 9,
    seed: Optional[int] = None
):
    """
    Analyze a batch of random STEAD samples.
    
    Args:
        model: Trained model
        stead_dataset: STEAD dataset
        device: Device to run on
        output_dir: Output directory
        n_samples: Number of samples
        seed: Random seed
    """
    logger.info(f"\nGenerating grid of {n_samples} random samples...")
    
    # Load random samples
    samples = load_random_samples(stead_dataset, n_samples=n_samples, seed=seed)
    
    if not samples:
        logger.warning("No samples loaded!")
        return
    
    # Get predictions for all samples
    analysis_samples = []
    for sample in samples:
        waveform = sample['waveform']
        if waveform.shape[0] != 3:
            waveform = waveform.T
        
        # Get model prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(device)
            logits = model(input_tensor)
            prediction = torch.sigmoid(logits).item()
        
        analysis_samples.append({
            'waveform': waveform,
            'prediction': prediction,
            'label': sample['label'],
            'p_arrival_sample': sample.get('p_arrival_sample'),
            'event_id': f"Sample_{len(analysis_samples):03d}"
        })
    
    # Plot grid
    grid_size = int(np.ceil(np.sqrt(len(analysis_samples))))
    fig = plot_grid_of_samples(
        analysis_samples,
        grid_shape=(grid_size, grid_size),
        sampling_rate=100.0,
        title="STEAD Dataset Analysis: Random Sample Predictions",
        save_path=output_dir / "grid_random_samples.png"
    )
    
    logger.info(f"✓ Saved grid with {len(analysis_samples)} samples")


def analyze_earthquakes_vs_noise(
    model,
    stead_dataset,
    device: str,
    output_dir: Path,
    n_events: int = 5,
    n_noise: int = 5,
    seed: Optional[int] = None
):
    """
    Analyze earthquakes vs noise side-by-side.
    
    Args:
        model: Trained model
        stead_dataset: STEAD dataset
        device: Device to run on
        output_dir: Output directory
        n_events: Number of earthquake samples
        n_noise: Number of noise samples
        seed: Random seed
    """
    logger.info(f"\nAnalyzing {n_events} earthquakes vs {n_noise} noise samples...")
    
    try:
        from eew.visualization import load_balanced_samples
        event_samples, noise_samples = load_balanced_samples(
            stead_dataset,
            n_events=n_events,
            n_noise=n_noise,
            seed=seed
        )
    except Exception as e:
        logger.warning(f"Could not load balanced samples: {e}")
        return
    
    # Prepare analysis
    all_samples = []
    
    # Process earthquakes
    for i, sample in enumerate(event_samples):
        waveform = sample['waveform']
        if waveform.shape[0] != 3:
            waveform = waveform.T
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(device)
            logits = model(input_tensor)
            prediction = torch.sigmoid(logits).item()
        
        all_samples.append({
            'waveform': waveform,
            'prediction': prediction,
            'label': 1,
            'p_arrival_sample': sample.get('p_arrival_sample'),
            'event_id': f"Event_{i:02d}"
        })
    
    # Process noise
    for i, sample in enumerate(noise_samples):
        waveform = sample['waveform']
        if waveform.shape[0] != 3:
            waveform = waveform.T
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(device)
            logits = model(input_tensor)
            prediction = torch.sigmoid(logits).item()
        
        all_samples.append({
            'waveform': waveform,
            'prediction': prediction,
            'label': 0,
            'p_arrival_sample': sample.get('p_arrival_sample'),
            'event_id': f"Noise_{i:02d}"
        })
    
    # Plot grid
    n_total = len(all_samples)
    grid_size = int(np.ceil(np.sqrt(n_total)))
    fig = plot_grid_of_samples(
        all_samples,
        grid_shape=(grid_size, grid_size),
        sampling_rate=100.0,
        title=f"STEAD Analysis: {len(event_samples)} Earthquakes vs {len(noise_samples)} Noise",
        save_path=output_dir / "grid_earthquakes_vs_noise.png"
    )
    
    logger.info(f"✓ Saved grid with {len(event_samples)} earthquakes and {len(noise_samples)} noise samples")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Advanced STEAD Real Data Visualization"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='results_improved_v3/checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--stead-path',
        type=str,
        default=None,
        help='Path to STEAD data or seisbench dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures/stead_analysis',
        help='Directory to save figures'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['demo', 'random', 'balanced', 'all'],
        default='demo',
        help='Analysis mode'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=9,
        help='Number of samples for grid analysis'
    )
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1
    
    # Load model
    model = load_model(str(checkpoint_path), device=args.device)
    
    logger.info("\n" + "="*70)
    logger.info("STEAD REAL DATA ANALYSIS")
    logger.info("="*70)
    
    # Check if STEAD is available
    stead_dataset = None
    try:
        if args.stead_path:
            from eew.data_loader import STEADLoader
            loader = STEADLoader(data_path=args.stead_path, region='chile')
            stead_dataset = loader.load_dataset()
        else:
            try:
                import seisbench
                import seisbench.data as sbd
                logger.info("Downloading STEAD dataset from seisbench...")
                stead_dataset = sbd.STEAD()
            except Exception as e:
                logger.warning(f"Could not load STEAD: {e}")
    except Exception as e:
        logger.warning(f"STEAD not available: {e}")
    
    # Run analysis based on mode
    if args.mode == 'demo' or stead_dataset is None:
        logger.info("Running synthetic demo...")
        
        # Import synthetic demo
        from visualize_eew import (
            demo_1_full_trace,
            demo_2_realtime_simulation,
            demo_3_window_predictions,
            demo_4_grid_mosaic,
            demo_5_detection_comparison,
        )
        
        demos = [
            demo_1_full_trace,
            demo_2_realtime_simulation,
            demo_3_window_predictions,
            demo_4_grid_mosaic,
            demo_5_detection_comparison,
        ]
        
        for demo in demos:
            try:
                demo(model, args.device, output_dir)
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
    
    elif stead_dataset is not None:
        if args.mode in ['random', 'all']:
            analyze_random_batch(
                model,
                stead_dataset,
                args.device,
                output_dir,
                n_samples=args.n_samples
            )
        
        if args.mode in ['balanced', 'all']:
            analyze_earthquakes_vs_noise(
                model,
                stead_dataset,
                args.device,
                output_dir,
                n_events=args.n_samples // 2,
                n_noise=args.n_samples // 2
            )
    
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETED!")
    logger.info(f"Figures saved to: {output_dir}")
    logger.info("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
