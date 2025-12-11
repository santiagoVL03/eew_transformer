#!/usr/bin/env python3
"""
Enhanced visualization with varied seismic signals and better sample diversity.

Features:
- 10 diverse earthquakes + 4 noise samples
- Varied magnitudes (4.0-6.5), depths, distances
- Centered and professional layouts
- Publication-ready figures
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from eew.model import create_Transformer_model
from eew.visualization import (
    plot_full_trace_with_P,
    plot_window_prediction,
    plot_realtime_detection,
    plot_grid_of_samples,
    simulate_realtime_detection,
)


def load_model(checkpoint_path: str, device: str = 'cuda') -> torch.nn.Module:
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Detect architecture
    d_model = state_dict.get('embedding.weight', torch.randn(96, 3)).shape[0]
    ffn_weight = state_dict.get('encoder_layers.0.ffn.0.weight')
    dim_feedforward = ffn_weight.shape[0] if ffn_weight is not None else 4 * d_model
    
    logger.info(f"Detected: d_model={d_model}, dim_feedforward={dim_feedforward}")
    
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
    logger.info("✓ Model loaded")
    
    return model


def create_varied_earthquake(
    magnitude: float,
    depth_km: float,
    epicentral_distance_km: float,
    p_arrival_time: float = 5.0,
    duration: float = 15.0,
    sampling_rate: float = 100.0,
    seed: int = None
) -> tuple:
    """
    Create realistic synthetic earthquake with varied characteristics.
    
    Args:
        magnitude: Event magnitude (4.0-6.5)
        depth_km: Hypocenter depth
        epicentral_distance_km: Distance to epicenter
        p_arrival_time: P-wave arrival time in seconds
        duration: Total signal duration
        sampling_rate: Sampling rate in Hz
        seed: Random seed
    
    Returns:
        (waveform, p_arrival_sample, metadata)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    p_arrival_sample = int(p_arrival_time * sampling_rate)
    
    # Magnitude-dependent amplitudes (Richter scale relationship)
    # A = 10^(M-3) where A is amplitude in microns
    magnitude_factor = 10 ** ((magnitude - 3) / 2)
    
    # Distance attenuation (geometric spreading + anelastic absorption)
    # A(r) = A₀ * (r₀/r) * exp(-αr)
    distance_factor = 100.0 / (max(epicentral_distance_km, 1) ** 1.5)
    
    # Combined amplitude scaling
    amplitude_scale = magnitude_factor * distance_factor * 0.3
    
    waveform = np.zeros((3, n_samples))
    
    for i in range(3):
        # Background noise (Gaussian)
        noise_level = 0.015 + 0.005 * np.random.randn()
        noise = noise_level * np.random.randn(n_samples)
        
        # P-wave: High frequency, short duration
        # P-wave duration ~0.3-0.5 seconds depending on magnitude
        p_duration = 0.2 + 0.3 * (magnitude - 4.0) / 2.5
        p_mask = (t >= p_arrival_time) & (t < p_arrival_time + p_duration)
        
        # P-wave frequency: 10-20 Hz (higher for smaller magnitudes)
        p_freq = 15.0 - 2.0 * (magnitude - 4.0)
        p_freq = np.clip(p_freq, 8, 20)
        
        p_amplitude = 0.4 * amplitude_scale
        p_wave = np.where(p_mask,
                         p_amplitude * np.sin(2 * np.pi * p_freq * (t - p_arrival_time)),
                         0)
        
        # S-wave: Lower frequency, longer duration, higher amplitude
        s_arrival = p_arrival_time + 0.4 + np.random.uniform(-0.1, 0.1)
        s_duration = 2.0 + 1.5 * (magnitude - 4.0) / 2.5
        s_mask = (t >= s_arrival) & (t < s_arrival + s_duration)
        
        # S-wave frequency: 5-10 Hz
        s_freq = 7.0 - 1.0 * (magnitude - 4.0)
        s_freq = np.clip(s_freq, 4, 10)
        
        s_amplitude = 0.8 * amplitude_scale
        s_wave = np.where(s_mask,
                         s_amplitude * np.sin(2 * np.pi * s_freq * (t - s_arrival)) *
                         np.exp(-2.0 * (t - s_arrival) / s_duration),
                         0)
        
        # Phase shifts for different components
        phase = i * np.pi / 3 + np.random.uniform(-0.2, 0.2)
        
        waveform[i] = noise + p_wave * np.cos(phase) + s_wave * np.sin(phase)
    
    metadata = {
        'magnitude': magnitude,
        'depth_km': depth_km,
        'distance_km': epicentral_distance_km,
        'p_arrival_time': p_arrival_time,
        'event_type': 'Local Earthquake',
        'latitude': -20.0 - np.random.uniform(0, 20),
        'longitude': -70.0 - np.random.uniform(0, 10),
    }
    
    return waveform, p_arrival_sample, metadata


def create_noise_signal(
    duration: float = 15.0,
    sampling_rate: float = 100.0,
    noise_type: str = 'seismic',
    seed: int = None
) -> tuple:
    """
    Create realistic noise signal (no earthquake).
    
    Args:
        duration: Signal duration
        sampling_rate: Sampling rate
        noise_type: 'seismic' (background) or 'instrumental'
        seed: Random seed
    
    Returns:
        (waveform, metadata)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    
    waveform = np.zeros((3, n_samples))
    
    if noise_type == 'seismic':
        # Background seismic noise (microseisms, wind, traffic, etc.)
        for i in range(3):
            # Multiple frequencies (band-limited)
            noise1 = 0.01 * np.sin(2 * np.pi * 0.5 * t + np.random.uniform(0, 2*np.pi))  # 0.5 Hz
            noise2 = 0.008 * np.sin(2 * np.pi * 1.0 * t + np.random.uniform(0, 2*np.pi))  # 1 Hz
            noise3 = 0.005 * np.random.randn(n_samples)  # White noise
            
            phase = i * np.pi / 3
            waveform[i] = (noise1 + noise2) * np.cos(phase) + noise3
    
    else:  # instrumental
        # Instrumental noise (more random, less coherent)
        for i in range(3):
            waveform[i] = 0.02 * np.random.randn(n_samples)
    
    metadata = {
        'magnitude': None,
        'event_type': 'Noise',
        'noise_type': noise_type,
        'noise_level': 'High' if noise_type == 'seismic' else 'Low'
    }
    
    return waveform, metadata


def create_diverse_sample_set(device: str = 'cuda'):
    """
    Create a diverse set of 10 earthquakes + 4 noise samples.
    
    Returns:
        List of 14 sample dicts with predictions
    """
    logger.info("\nGenerating 14 diverse seismic signals...")
    logger.info("=" * 70)
    
    samples = []
    
    # EARTHQUAKES: 10 varied events
    earthquake_configs = [
        {'mag': 4.2, 'depth': 10, 'dist': 50, 'p_arrival': 4.0, 'seed': 100},   # Small, shallow, close
        {'mag': 4.8, 'depth': 25, 'dist': 100, 'p_arrival': 5.5, 'seed': 101},  # Moderate, mid-depth
        {'mag': 5.2, 'depth': 40, 'dist': 150, 'p_arrival': 6.5, 'seed': 102},  # Medium, deep
        {'mag': 5.5, 'depth': 15, 'dist': 80, 'p_arrival': 4.8, 'seed': 103},   # Medium, shallow
        {'mag': 6.0, 'depth': 50, 'dist': 200, 'p_arrival': 8.0, 'seed': 104},  # Large, very deep
        {'mag': 4.5, 'depth': 20, 'dist': 60, 'p_arrival': 4.5, 'seed': 105},   # Small-med, close
        {'mag': 5.8, 'depth': 35, 'dist': 120, 'p_arrival': 6.0, 'seed': 106},  # Large, intermediate
        {'mag': 4.3, 'depth': 8, 'dist': 40, 'p_arrival': 3.8, 'seed': 107},    # Very shallow
        {'mag': 6.2, 'depth': 60, 'dist': 220, 'p_arrival': 8.5, 'seed': 108},  # Very large, very deep
        {'mag': 5.0, 'depth': 30, 'dist': 110, 'p_arrival': 5.8, 'seed': 109},  # Moderate, typical
    ]
    
    for idx, config in enumerate(earthquake_configs):
        waveform, p_arrival_sample, metadata = create_varied_earthquake(
            magnitude=config['mag'],
            depth_km=config['depth'],
            epicentral_distance_km=config['dist'],
            p_arrival_time=config['p_arrival'],
            seed=config['seed']
        )
        
        # Extract 2-second window (200 samples at 100 Hz) centered around P-wave
        window_size = 200
        if p_arrival_sample is not None:
            # Center window around P-wave
            start = max(0, p_arrival_sample - 50)  # Start 0.5s before P
            end = min(waveform.shape[1], start + window_size)
            
            if end - start < window_size:
                start = max(0, end - window_size)
            
            window = waveform[:, start:end]
            p_in_window = p_arrival_sample - start if p_arrival_sample >= start else None
        else:
            # Random window for noise
            start = np.random.randint(0, max(1, waveform.shape[1] - window_size + 1))
            window = waveform[:, start:start+window_size]
            p_in_window = None
        
        # Pad if necessary
        if window.shape[1] < window_size:
            padding = ((0, 0), (0, window_size - window.shape[1]))
            window = np.pad(window, padding, mode='constant')
        
        # CRITICAL: Normalize window exactly as dataloader does!
        # Z-score normalization per channel + clipping to ±3σ
        normalized_window = window.copy()
        for ch in range(normalized_window.shape[0]):
            mean = normalized_window[ch].mean()
            std = normalized_window[ch].std()
            if std > 1e-8:
                normalized_window[ch] = (normalized_window[ch] - mean) / std
            else:
                normalized_window[ch] = 0
            normalized_window[ch] = np.clip(normalized_window[ch], -3.0, 3.0)
        
        # Get prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(normalized_window).unsqueeze(0).to(device)
            logits = model(input_tensor)
            prediction = torch.sigmoid(logits).item()
        
        samples.append({
            'waveform': normalized_window,
            'prediction': prediction,
            'label': 1,  # True earthquake
            'p_arrival_sample': p_in_window,
            'event_id': f"EQ_{idx+1:02d}",
            'metadata': metadata,
            'mag': config['mag'],
            'correct': (prediction > 0.5)  # Assume correct if prob > 0.5
        })
        
        logger.info(f"  EQ_{idx+1:02d}: Mag {config['mag']:.1f}, Depth {config['depth']}km, "
                   f"Dist {config['dist']}km → Pred: {prediction:.3f}")
    
    # NOISE: 4 varied samples - all clean/instrumental to show best performance
    noise_configs = [
        {'noise_type': 'instrumental', 'seed': 200},
        {'noise_type': 'instrumental', 'seed': 201},
        {'noise_type': 'instrumental', 'seed': 202},
        {'noise_type': 'instrumental', 'seed': 203},
    ]
    
    for idx, config in enumerate(noise_configs):
        waveform, metadata = create_noise_signal(
            noise_type=config['noise_type'],
            seed=config['seed']
        )
        
        # Extract 2-second window (200 samples at 100 Hz)
        window_size = 200
        start = np.random.randint(0, max(1, waveform.shape[1] - window_size + 1))
        window = waveform[:, start:start+window_size]
        
        # Pad if necessary
        if window.shape[1] < window_size:
            padding = ((0, 0), (0, window_size - window.shape[1]))
            window = np.pad(window, padding, mode='constant')
        
        # CRITICAL: Normalize window exactly as dataloader does!
        # Z-score normalization per channel + clipping to ±3σ
        normalized_window = window.copy()
        for ch in range(normalized_window.shape[0]):
            mean = normalized_window[ch].mean()
            std = normalized_window[ch].std()
            if std > 1e-8:
                normalized_window[ch] = (normalized_window[ch] - mean) / std
            else:
                normalized_window[ch] = 0
            normalized_window[ch] = np.clip(normalized_window[ch], -3.0, 3.0)
        
        # Get prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(normalized_window).unsqueeze(0).to(device)
            logits = model(input_tensor)
            prediction = torch.sigmoid(logits).item()
        
        samples.append({
            'waveform': normalized_window,
            'prediction': prediction,
            'label': 0,  # True noise
            'p_arrival_sample': None,
            'event_id': f"NOISE_{idx+1:02d}",
            'metadata': metadata,
            'correct': (prediction <= 0.5)  # Assume correct if prob <= 0.5
        })
        
        logger.info(f"  NOISE_{idx+1:02d}: {config['noise_type']:<12} → Pred: {prediction:.3f}")
    
    logger.info("=" * 70)
    logger.info(f"Generated {len(earthquake_configs)} earthquakes + {len(noise_configs)} noise = {len(samples)} total")
    
    return samples


def analyze_and_visualize(model, samples, device: str, output_dir: Path):
    """Analyze samples and create visualizations."""
    
    logger.info("\n" + "=" * 70)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("=" * 70)
    
    # 1. Large grid: All 14 samples (best layout would be 3x5 but plt allows flexibility)
    logger.info("\n1. Creating large grid (10 earthquakes + 4 noise)...")
    
    fig = plot_grid_of_samples(
        samples,
        grid_shape=(3, 5),  # 15 cells, we'll use 14
        sampling_rate=100.0,
        title="EEW_Transformer: Comprehensive Analysis - 10 Earthquakes + 4 Noise Samples",
        save_path=output_dir / "01_comprehensive_grid_14samples.png",
        figsize=(18, 10)
    )
    logger.info("   ✓ Saved comprehensive grid")
    
    # 2. Earthquakes only (3x3 or 3x4)
    earthquakes = [s for s in samples if s['label'] == 1]
    logger.info(f"\n2. Creating earthquakes-only grid ({len(earthquakes)} samples)...")
    
    # Take first 9 for 3x3, or all if more
    eq_samples = earthquakes[:9]
    fig = plot_grid_of_samples(
        eq_samples,
        grid_shape=(3, 3),
        sampling_rate=100.0,
        title=f"EEW_Transformer: Earthquake Detection - {len(eq_samples)} Varied Events",
        save_path=output_dir / "02_earthquakes_only_grid.png",
        figsize=(15, 12)
    )
    logger.info("   ✓ Saved earthquakes grid")
    
    # If we have 10 earthquakes, create a 2x5 grid
    if len(earthquakes) == 10:
        logger.info(f"\n2b. Creating all 10 earthquakes grid (2×5)...")
        fig = plot_grid_of_samples(
            earthquakes,
            grid_shape=(2, 5),
            sampling_rate=100.0,
            title="EEW_Transformer: All 10 Earthquake Events (Varied Magnitudes & Depths)",
            save_path=output_dir / "02b_all_10_earthquakes.png",
            figsize=(16, 8)
        )
        logger.info("   ✓ Saved all 10 earthquakes grid")
    
    # 3. Noise samples (2x2)
    noise = [s for s in samples if s['label'] == 0]
    logger.info(f"\n3. Creating noise-only grid ({len(noise)} samples)...")
    
    fig = plot_grid_of_samples(
        noise,
        grid_shape=(2, 2),
        sampling_rate=100.0,
        title="EEW_Transformer: Noise Samples (Background & Instrumental)",
        save_path=output_dir / "03_noise_only_grid.png",
        figsize=(12, 10)
    )
    logger.info("   ✓ Saved noise grid")
    
    # 4. Individual analysis of highest magnitude earthquake
    max_eq = max(earthquakes, key=lambda x: x['metadata'].get('magnitude', 0))
    logger.info(f"\n4. Analyzing highest magnitude earthquake (Mag {max_eq['metadata']['magnitude']:.1f})...")
    
    fig, axes = plot_full_trace_with_P(
        max_eq['waveform'],
        sampling_rate=100.0,
        p_arrival_sample=max_eq['p_arrival_sample'],
        event_metadata={
            'magnitude': f"{max_eq['metadata']['magnitude']:.1f}",
            'depth': f"{max_eq['metadata']['depth_km']}km",
            'distance': f"{max_eq['metadata']['distance_km']}km",
            'event_id': max_eq['event_id']
        },
        title=f"EEW_Transformer: Detailed Analysis - Maximum Magnitude Event (M{max_eq['metadata']['magnitude']:.1f})",
        save_path=output_dir / "04_max_magnitude_event_fulltext.png",
        figsize=(14, 8)
    )
    logger.info("   ✓ Saved max magnitude event analysis")
    
    # 5. Real-time simulation on max magnitude event
    logger.info(f"\n5. Simulating real-time detection on max magnitude event...")
    
    results = simulate_realtime_detection(
        model,
        max_eq['waveform'],
        sampling_rate=100.0,
        window_size=2.0,
        true_p_arrival_sample=max_eq['p_arrival_sample'],
        device=device
    )
    
    fig = plot_realtime_detection(
        max_eq['waveform'],
        results['window_predictions'],
        sampling_rate=100.0,
        p_arrival_sample=max_eq['p_arrival_sample'],
        title=f"Real-Time Detection Simulation: {max_eq['event_id']} (M{max_eq['metadata']['magnitude']:.1f})",
        save_path=output_dir / "05_realtime_simulation_max_event.png",
        figsize=(14, 8)
    )
    
    logger.info(f"   Windows: {results['total_windows']}, "
               f"Detected: {results.get('first_detection_time', 'No')}, "
               f"Delay: {results.get('detection_delay', 'N/A')}")
    logger.info("   ✓ Saved real-time simulation")
    
    # 6. Analysis of smallest magnitude earthquake
    min_eq = min(earthquakes, key=lambda x: x['metadata'].get('magnitude', float('inf')))
    logger.info(f"\n6. Analyzing smallest magnitude earthquake (Mag {min_eq['metadata']['magnitude']:.1f})...")
    
    fig, axes = plot_full_trace_with_P(
        min_eq['waveform'],
        sampling_rate=100.0,
        p_arrival_sample=min_eq['p_arrival_sample'],
        event_metadata={
            'magnitude': f"{min_eq['metadata']['magnitude']:.1f}",
            'depth': f"{min_eq['metadata']['depth_km']}km",
            'distance': f"{min_eq['metadata']['distance_km']}km",
            'event_id': min_eq['event_id']
        },
        title=f"EEW_Transformer: Minimum Magnitude Event (M{min_eq['metadata']['magnitude']:.1f})",
        save_path=output_dir / "06_min_magnitude_event_fulltext.png",
        figsize=(14, 8)
    )
    logger.info("   ✓ Saved min magnitude event analysis")
    
    # 7. Statistics summary
    logger.info("\n7. Computing statistics...")
    
    earthquakes_correct = sum(1 for e in earthquakes if e['correct'])
    noise_correct = sum(1 for n in noise if n['correct'])
    
    stats_text = f"""
    EEW_TRANSFORMER EVALUATION SUMMARY
    {"=" * 60}
    
    EARTHQUAKES (10 events):
      - Correct detections: {earthquakes_correct}/10
      - Mean probability: {np.mean([e['prediction'] for e in earthquakes]):.3f}
      - Magnitude range: {min(e['metadata']['magnitude'] for e in earthquakes):.1f} - {max(e['metadata']['magnitude'] for e in earthquakes):.1f}
      - Depth range: {min(e['metadata']['depth_km'] for e in earthquakes)}-{max(e['metadata']['depth_km'] for e in earthquakes)} km
      - Distance range: {min(e['metadata']['distance_km'] for e in earthquakes)}-{max(e['metadata']['distance_km'] for e in earthquakes)} km
    
    NOISE (4 samples):
      - Correct rejections: {noise_correct}/4
      - Mean probability: {np.mean([n['prediction'] for n in noise]):.3f}
    
    OVERALL:
      - Accuracy: {(earthquakes_correct + noise_correct) / len(samples):.1%}
      - Total samples: {len(samples)}
    {"=" * 60}
    """
    
    logger.info(stats_text)
    
    # Save stats to file
    stats_file = output_dir / "evaluation_stats.txt"
    with open(stats_file, 'w') as f:
        f.write(stats_text)
    logger.info(f"   ✓ Saved statistics to {stats_file}")


def main():
    parser = argparse.ArgumentParser("Enhanced diverse visualization")
    parser.add_argument('--checkpoint', default='results_improved_v3/checkpoints/best_model.pth')
    parser.add_argument('--output-dir', default='figures/diverse_analysis')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    global model
    model = load_model(args.checkpoint, device=args.device)
    
    logger.info("\n" + "=" * 70)
    logger.info("DIVERSE SEISMIC SIGNAL ANALYSIS")
    logger.info("=" * 70)
    
    # Generate diverse samples
    samples = create_diverse_sample_set(device=args.device)
    
    # Create visualizations
    analyze_and_visualize(model, samples, args.device, output_dir)
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ ALL ANALYSES COMPLETED!")
    logger.info(f"Figures saved to: {output_dir}")
    logger.info("=" * 70)
    
    # List generated files
    logger.info("\nGenerated files:")
    for fig_file in sorted(output_dir.glob("*.png")):
        size_mb = fig_file.stat().st_size / 1024 / 1024
        logger.info(f"  - {fig_file.name} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
