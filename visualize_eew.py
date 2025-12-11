#!/usr/bin/env python3
"""
Complete demonstration script for EEW_Transformer visualization and analysis.

This script showcases all visualization capabilities:
1. Full trace visualization with P-wave markers
2. Real-time detection simulation
3. Individual window predictions
4. Grid mosaic of predictions
5. Comparison analysis

Usage:
    python visualize_eew.py --checkpoint path/to/checkpoint.pth \
                            --stead-path path/to/stead/data \
                            --sample-idx 0 \
                            --output-dir figures/
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Optional

# Configure logging
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


def create_synthetic_waveform(duration: float = 10.0, 
                             sampling_rate: float = 100.0,
                             p_arrival_time: float = 5.0) -> tuple:
    """
    Create a synthetic earthquake waveform for demonstration.
    
    Args:
        duration: Duration in seconds
        sampling_rate: Sampling rate in Hz
        p_arrival_time: Time of P-wave arrival in seconds
    
    Returns:
        (waveform, p_arrival_sample, metadata)
    """
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    p_arrival_sample = int(p_arrival_time * sampling_rate)
    
    waveform = np.zeros((3, n_samples))
    
    for i in range(3):
        # Background noise
        noise = 0.03 * np.random.randn(n_samples)
        
        # P-wave (high frequency, brief)
        p_mask = (t >= p_arrival_time) & (t < p_arrival_time + 0.5)
        p_wave = np.where(p_mask,
                         0.4 * np.sin(2 * np.pi * 12 * (t - p_arrival_time)),
                         0)
        
        # S-wave (lower frequency, longer duration)
        s_arrival = p_arrival_time + 0.8
        s_mask = t >= s_arrival
        s_wave = np.where(s_mask,
                         0.6 * np.sin(2 * np.pi * 5 * (t - s_arrival)) * 
                         np.exp(-1 * (t - s_arrival)),
                         0)
        
        # Add phase differences
        phase = i * np.pi / 3
        waveform[i] = noise + p_wave * np.cos(phase) + s_wave * np.sin(phase)
    
    metadata = {
        'magnitude': 5.2,
        'event_id': 'SYNTHETIC_001',
        'station': 'CH.VDLS',
        'latitude': -34.5,
        'longitude': -71.5,
    }
    
    return waveform, p_arrival_sample, metadata


def demo_1_full_trace(model, device: str, output_dir: Path):
    """Demo 1: Full trace visualization with P-wave marker."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 1: Full Trace Visualization with P-wave Arrival")
    logger.info("="*70)
    
    # Create synthetic waveform
    waveform, p_arrival_sample, metadata = create_synthetic_waveform(
        duration=20.0,
        p_arrival_time=8.0
    )
    
    # Plot
    fig, axes = plot_full_trace_with_P(
        waveform,
        sampling_rate=100.0,
        p_arrival_sample=p_arrival_sample,
        event_metadata=metadata,
        title="EEW_Transformer - Full Seismic Trace with P-wave Arrival",
        save_path=output_dir / "01_full_trace_with_p.png"
    )
    
    logger.info("✓ Full trace visualization saved")


def demo_2_realtime_simulation(model, device: str, output_dir: Path):
    """Demo 2: Real-time detection simulation."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 2: Real-Time Detection Simulation")
    logger.info("="*70)
    
    # Create synthetic waveform
    waveform, p_arrival_sample, metadata = create_synthetic_waveform(
        duration=20.0,
        p_arrival_time=8.0
    )
    
    # Simulate real-time detection
    logger.info("Processing windows sequentially...")
    results = simulate_realtime_detection(
        model,
        waveform,
        sampling_rate=100.0,
        window_size=2.0,
        true_p_arrival_sample=p_arrival_sample,
        detection_threshold=0.5,
        device=device
    )
    
    # Print statistics
    stats = get_detection_statistics(results)
    logger.info(f"Total windows processed: {stats['total_windows']}")
    logger.info(f"Mean probability: {stats['mean_probability']:.3f}")
    logger.info(f"Detection rate: {stats['detection_rate']:.1%}")
    
    if 'first_detection_time' in stats:
        logger.info(f"First detection at: {stats['first_detection_time']:.2f}s")
        logger.info(f"True P-wave at: {stats['true_p_arrival_time']:.2f}s")
        logger.info(f"Detection delay: {stats['detection_delay']:+.2f}s")
    
    # Plot real-time detection
    fig = plot_realtime_detection(
        waveform,
        results['window_predictions'],
        sampling_rate=100.0,
        p_arrival_sample=p_arrival_sample,
        title="Real-Time Detection Simulation: Window-by-Window Processing",
        save_path=output_dir / "02_realtime_detection.png"
    )
    
    logger.info("✓ Real-time detection visualization saved")


def demo_3_window_predictions(model, device: str, output_dir: Path):
    """Demo 3: Individual window predictions."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 3: Individual Window Predictions")
    logger.info("="*70)
    
    # Create synthetic waveform
    waveform, p_arrival_sample, _ = create_synthetic_waveform(
        duration=10.0,
        p_arrival_time=5.0
    )
    
    # Extract several windows and make predictions
    window_size = 200  # 2 seconds at 100 Hz
    window_positions = [
        (300, "Before P-wave"),
        (400, "At P-wave arrival"),
        (500, "After P-wave (S-wave)"),
    ]
    
    for start_sample, description in window_positions:
        if start_sample + window_size <= waveform.shape[1]:
            window = waveform[:, start_sample:start_sample + window_size]
            
            # Get model prediction
            with torch.no_grad():
                input_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
                logits = model(input_tensor)
                prediction = torch.sigmoid(logits).item()
            
            # Determine if P-wave is in window
            p_in_window = None
            if start_sample <= p_arrival_sample < start_sample + window_size:
                p_in_window = p_arrival_sample - start_sample
            
            # Determine true label
            true_label = 1 if p_in_window is not None else 0
            
            # Plot
            fig = plot_window_prediction(
                window,
                prediction,
                true_label,
                sampling_rate=100.0,
                p_arrival_sample=p_in_window,
                window_idx=start_sample // window_size,
                title=f"Window at sample {start_sample}: {description}",
                save_path=output_dir / f"03_window_{start_sample:05d}.png"
            )
            
            logger.info(f"  ✓ Window at sample {start_sample}: Prediction={prediction:.3f}, Label={'Event' if true_label else 'Noise'}")


def demo_4_grid_mosaic(model, device: str, output_dir: Path):
    """Demo 4: Grid mosaic of predictions."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 4: Grid Mosaic of Predictions")
    logger.info("="*70)
    
    # Create multiple synthetic samples
    samples = []
    
    for i in range(9):
        if i < 5:
            # Events
            p_arrival_time = 0.8 + np.random.randn() * 0.3
            p_arrival_time = np.clip(p_arrival_time, 0.5, 1.5)
        else:
            # Noise (no P-wave)
            p_arrival_time = None
        
        waveform, p_arrival_sample, _ = create_synthetic_waveform(
            duration=2.0,
            p_arrival_time=p_arrival_time if p_arrival_time else 1.0
        )
        
        # Get prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(device)
            logits = model(input_tensor)
            prediction = torch.sigmoid(logits).item()
        
        # True label
        true_label = 1 if p_arrival_time is not None else 0
        
        samples.append({
            'waveform': waveform,
            'prediction': prediction,
            'label': true_label,
            'p_arrival_sample': p_arrival_sample if p_arrival_time else None,
            'event_id': f'SYNTHETIC_{i:03d}'
        })
    
    # Plot grid
    fig = plot_grid_of_samples(
        samples,
        grid_shape=(3, 3),
        sampling_rate=100.0,
        title="EEW_Transformer: 3×3 Grid of Predictions (Mixed Events and Noise)",
        save_path=output_dir / "04_grid_mosaic.png"
    )
    
    logger.info(f"✓ Grid mosaic with {len(samples)} samples saved")


def demo_5_detection_comparison(model, device: str, output_dir: Path):
    """Demo 5: True vs predicted detection comparison."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 5: True vs Predicted P-wave Detection Comparison")
    logger.info("="*70)
    
    # Create synthetic waveform
    waveform, true_p_arrival, _ = create_synthetic_waveform(
        duration=15.0,
        p_arrival_time=6.5
    )
    
    # Simulate real-time detection to get predicted detections
    results = simulate_realtime_detection(
        model,
        waveform,
        sampling_rate=100.0,
        window_size=2.0,
        true_p_arrival_sample=true_p_arrival,
        device=device
    )
    
    # Get detected window positions
    predicted_detections = [
        pred['start_sample'] 
        for pred in results['window_predictions']
        if pred['prediction'] > 0.5
    ]
    
    # Plot comparison
    fig = plot_detection_comparison(
        waveform,
        true_p_arrival,
        predicted_detections,
        sampling_rate=100.0,
        title="Comparison: True P-wave Arrival vs Model Detections",
        save_path=output_dir / "05_detection_comparison.png"
    )
    
    logger.info(f"✓ Detection comparison saved")
    if predicted_detections:
        logger.info(f"  Model detected P-wave at sample: {predicted_detections[0]}")
        logger.info(f"  True P-wave at sample: {true_p_arrival}")
        delay = (predicted_detections[0] - true_p_arrival) / 100.0
        logger.info(f"  Detection delay: {delay:+.2f}s")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="EEW_Transformer Visualization Demo"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='results_improved_v3/checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures',
        help='Directory to save figures'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on'
    )
    parser.add_argument(
        '--demo',
        type=int,
        default=None,
        help='Run specific demo (1-5), or all if not specified'
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
    logger.info("EEW_TRANSFORMER VISUALIZATION DEMONSTRATION")
    logger.info("="*70)
    
    # Run demos
    demos = {
        1: demo_1_full_trace,
        2: demo_2_realtime_simulation,
        3: demo_3_window_predictions,
        4: demo_4_grid_mosaic,
        5: demo_5_detection_comparison,
    }
    
    if args.demo is not None:
        if args.demo in demos:
            try:
                demos[args.demo](model, args.device, output_dir)
            except Exception as e:
                logger.error(f"Error in demo {args.demo}: {e}", exc_info=True)
        else:
            logger.error(f"Invalid demo number: {args.demo}")
    else:
        # Run all demos
        for demo_num in sorted(demos.keys()):
            try:
                demos[demo_num](model, args.device, output_dir)
            except Exception as e:
                logger.error(f"Error in demo {demo_num}: {e}", exc_info=True)
    
    logger.info("\n" + "="*70)
    logger.info("ALL DEMONSTRATIONS COMPLETED!")
    logger.info(f"Figures saved to: {output_dir}")
    logger.info("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
