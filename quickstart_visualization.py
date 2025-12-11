#!/usr/bin/env python3
"""
Quick start example: Load model, visualize predictions, generate figures.

Minimal example showing how to use the visualization module.
"""

import torch
import numpy as np
from pathlib import Path
import argparse

from eew.model import create_Transformer_model
from eew.visualization import (
    plot_full_trace_with_P,
    plot_window_prediction,
    plot_realtime_detection,
    plot_grid_of_samples,
    simulate_realtime_detection,
)


def main():
    parser = argparse.ArgumentParser("Quick start visualization example")
    parser.add_argument('--checkpoint', default='results_improved_v3/checkpoints/best_model.pth')
    parser.add_argument('--output-dir', default='figures')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Detect model architecture from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Get d_model from state dict
    embedding_weight_shape = state_dict.get('embedding.weight', torch.randn(96, 3)).shape
    d_model = embedding_weight_shape[0]
    
    # Get dim_feedforward from FFN weights
    ffn_weight = state_dict.get('encoder_layers.0.ffn.0.weight')
    if ffn_weight is not None:
        dim_feedforward = ffn_weight.shape[0]
    else:
        dim_feedforward = 4 * d_model  # Default: 4x d_model
    
    print(f"  Detected model: d_model={d_model}, dim_feedforward={dim_feedforward}")
    
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
    
    model.to(args.device)
    model.eval()
    print("✓ Model loaded")
    
    # Create synthetic waveform
    print("\nGenerating synthetic seismic waveform...")
    duration = 15.0
    sampling_rate = 100.0
    p_arrival_time = 6.0
    
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    p_arrival_sample = int(p_arrival_time * sampling_rate)
    
    waveform = np.zeros((3, n_samples))
    for i in range(3):
        noise = 0.02 * np.random.randn(n_samples)
        p_mask = (t >= p_arrival_time) & (t < p_arrival_time + 0.3)
        p_wave = np.where(p_mask, 0.35 * np.sin(2 * np.pi * 15 * (t - p_arrival_time)), 0)
        s_arrival = p_arrival_time + 0.5
        s_mask = t >= s_arrival
        s_wave = np.where(s_mask, 0.7 * np.sin(2 * np.pi * 6 * (t - s_arrival)) * np.exp(-0.8 * (t - s_arrival)), 0)
        phase = i * np.pi / 3
        waveform[i] = noise + p_wave * np.cos(phase) + s_wave * np.sin(phase)
    
    print("✓ Waveform created")
    
    # Example 1: Full trace
    print("\n1. Plotting full trace with P-wave marker...")
    fig, ax = plot_full_trace_with_P(
        waveform,
        sampling_rate=sampling_rate,
        p_arrival_sample=p_arrival_sample,
        event_metadata={'magnitude': 5.5, 'station': 'CH.VDLS', 'event_id': 'TEST_001'},
        save_path=output_dir / '01_full_trace.png'
    )
    print(f"   ✓ Saved to {output_dir / '01_full_trace.png'}")
    
    # Example 2: Real-time simulation
    print("\n2. Simulating real-time detection (window-by-window)...")
    results = simulate_realtime_detection(
        model,
        waveform,
        sampling_rate=sampling_rate,
        window_size=2.0,
        true_p_arrival_sample=p_arrival_sample,
        device=args.device
    )
    
    print(f"   Total windows: {results['total_windows']}")
    print(f"   True P-wave: {results['true_p_arrival_time']:.2f}s")
    if results['detected']:
        print(f"   Detected at: {results['first_detection_time']:.2f}s")
        print(f"   Detection delay: {results['detection_delay']:+.2f}s")
    else:
        print("   ✗ No detection")
    
    fig = plot_realtime_detection(
        waveform,
        results['window_predictions'],
        sampling_rate=sampling_rate,
        p_arrival_sample=p_arrival_sample,
        save_path=output_dir / '02_realtime_detection.png'
    )
    print(f"   ✓ Saved to {output_dir / '02_realtime_detection.png'}")
    
    # Example 3: Window predictions
    print("\n3. Plotting individual window predictions...")
    window_positions = [300, 400, 600]
    window_size_samples = 200
    
    for idx, start_sample in enumerate(window_positions):
        if start_sample + window_size_samples <= waveform.shape[1]:
            window = waveform[:, start_sample:start_sample + window_size_samples]
            
            with torch.no_grad():
                input_tensor = torch.FloatTensor(window).unsqueeze(0).to(args.device)
                logits = model(input_tensor)
                prediction = torch.sigmoid(logits).item()
            
            p_in_window = None
            true_label = 0
            if start_sample <= p_arrival_sample < start_sample + window_size_samples:
                p_in_window = p_arrival_sample - start_sample
                true_label = 1
            
            fig = plot_window_prediction(
                window,
                prediction,
                true_label,
                sampling_rate=sampling_rate,
                p_arrival_sample=p_in_window,
                window_idx=start_sample // window_size_samples,
                save_path=output_dir / f'03_window_{idx:02d}.png'
            )
            print(f"   ✓ Window {idx}: pred={prediction:.3f}, label={'EVENT' if true_label else 'NOISE'}")
    
    # Example 4: Grid of multiple samples
    print("\n4. Creating grid mosaic of mixed samples...")
    grid_samples = []
    
    for i in range(9):
        # Random events and noise
        if i < 5:
            p_time = 0.8 + 0.4 * np.random.randn()
            p_time = np.clip(p_time, 0.3, 1.7)
        else:
            p_time = None
        
        # Create sample waveform
        n_samp = int(2.0 * sampling_rate)
        t_samp = np.linspace(0, 2.0, n_samp)
        wf = np.zeros((3, n_samp))
        
        for ch in range(3):
            wf[ch] = 0.02 * np.random.randn(n_samp)
            if p_time:
                p_mask = (t_samp >= p_time) & (t_samp < p_time + 0.2)
                wf[ch] += np.where(p_mask, 0.4 * np.sin(2 * np.pi * 12 * (t_samp - p_time)), 0)
        
        # Get prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(wf).unsqueeze(0).to(args.device)
            logits = model(input_tensor)
            pred = torch.sigmoid(logits).item()
        
        label = 1 if p_time else 0
        p_samp = int(p_time * sampling_rate) if p_time else None
        
        grid_samples.append({
            'waveform': wf,
            'prediction': pred,
            'label': label,
            'p_arrival_sample': p_samp,
            'event_id': f'Sample_{i:03d}'
        })
    
    fig = plot_grid_of_samples(
        grid_samples,
        grid_shape=(3, 3),
        sampling_rate=sampling_rate,
        save_path=output_dir / '04_grid_mosaic.png'
    )
    print(f"   ✓ Saved to {output_dir / '04_grid_mosaic.png'}")
    
    print("\n" + "="*70)
    print("✓ ALL EXAMPLES COMPLETED!")
    print(f"Figures saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
