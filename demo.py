"""
Demo script showing how to use a trained Transformer model for inference.

Usage:
    python demo.py --checkpoint results/chile_2s/checkpoints/best_model.pth
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from eew.model import create_Transformer_model
from eew.preprocessing import preprocess_waveform


def load_model(checkpoint_path, device='cuda'):
    """
    Load trained Transformer model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model (default parameters)
    model = create_Transformer_model(
        seq_len=200,
        input_channels=3,
        d_model=64,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        classifier_hidden=200
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Trained at epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'metrics' in checkpoint:
        print(f"Validation metrics: {checkpoint['metrics']}")
    
    return model


def create_synthetic_earthquake(duration=2.0, sampling_rate=100):
    """
    Create a synthetic earthquake waveform for demonstration.
    
    Args:
        duration: Duration in seconds
        sampling_rate: Sampling rate in Hz
    
    Returns:
        3-channel waveform (3, n_samples)
    """
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Create synthetic P-wave arrival at t=1.0s
    p_arrival = 1.0
    
    waveform = np.zeros((3, n_samples))
    
    for i in range(3):
        # Background noise
        noise = 0.05 * np.random.randn(n_samples)
        
        # P-wave (high frequency, arrives first)
        p_mask = (t >= p_arrival) & (t < p_arrival + 0.2)
        p_wave = np.where(p_mask, 
                         0.3 * np.sin(2 * np.pi * 15 * (t - p_arrival)),
                         0)
        
        # S-wave (lower frequency, higher amplitude, arrives later)
        s_arrival = p_arrival + 0.3
        s_mask = t >= s_arrival
        s_wave = np.where(s_mask,
                         0.8 * np.sin(2 * np.pi * 8 * (t - s_arrival)) * 
                         np.exp(-2 * (t - s_arrival)),
                         0)
        
        # Phase shift for different components
        phase = i * np.pi / 3
        waveform[i] = noise + p_wave * np.cos(phase) + s_wave * np.sin(phase)
    
    return waveform


def create_synthetic_noise(duration=2.0, sampling_rate=100):
    """
    Create a synthetic noise waveform.
    
    Args:
        duration: Duration in seconds
        sampling_rate: Sampling rate in Hz
    
    Returns:
        3-channel waveform (3, n_samples)
    """
    n_samples = int(duration * sampling_rate)
    waveform = 0.1 * np.random.randn(3, n_samples)
    
    # Add some low-frequency variations
    t = np.linspace(0, duration, n_samples)
    for i in range(3):
        waveform[i] += 0.05 * np.sin(2 * np.pi * 0.5 * t + i)
    
    return waveform


def predict(model, waveform, device='cuda', use_mc_dropout=True, n_passes=10):
    """
    Make prediction on waveform.
    
    Args:
        model: Transformer model
        waveform: Input waveform (3, n_samples)
        device: Device
        use_mc_dropout: Use MC Dropout for uncertainty
        n_passes: Number of MC dropout passes
    
    Returns:
        probability, uncertainty
    """
    # Convert to tensor
    waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(device)
    
    if use_mc_dropout:
        # MC Dropout prediction
        mean_prob, std_prob, entropy = model.predict_with_uncertainty(
            waveform_tensor, n_passes=n_passes
        )
        probability = mean_prob.item()
        uncertainty = entropy.item()
    else:
        # Standard prediction
        with torch.no_grad():
            logit = model(waveform_tensor)
            probability = torch.sigmoid(logit).item()
            uncertainty = 0.0
    
    return probability, uncertainty


def plot_waveform_and_prediction(waveform, probability, uncertainty, title, save_path=None):
    """
    Plot waveform and prediction.
    
    Args:
        waveform: 3-channel waveform (3, n_samples)
        probability: Predicted probability
        uncertainty: Uncertainty (entropy)
        title: Plot title
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    sampling_rate = 100
    duration = waveform.shape[1] / sampling_rate
    t = np.linspace(0, duration, waveform.shape[1])
    
    channels = ['X (East)', 'Y (North)', 'Z (Vertical)']
    
    # Plot each channel
    for i in range(3):
        axes[i].plot(t, waveform[i], 'k-', linewidth=0.8)
        axes[i].set_ylabel(f'{channels[i]}', fontsize=11)
        axes[i].grid(alpha=0.3)
        axes[i].set_xlim(0, duration)
    
    axes[2].set_xlabel('Time (s)', fontsize=12)
    
    # Prediction panel
    axes[3].axis('off')
    
    # Determine prediction
    is_earthquake = probability > 0.5
    prediction_text = "EARTHQUAKE" if is_earthquake else "NOISE"
    color = 'red' if is_earthquake else 'green'
    
    # Display results
    axes[3].text(0.5, 0.7, f"Prediction: {prediction_text}", 
                ha='center', va='center', fontsize=20, weight='bold', color=color,
                transform=axes[3].transAxes)
    
    axes[3].text(0.5, 0.45, f"Probability: {probability:.3f}", 
                ha='center', va='center', fontsize=16,
                transform=axes[3].transAxes)
    
    axes[3].text(0.5, 0.25, f"Uncertainty: {uncertainty:.3f}", 
                ha='center', va='center', fontsize=14, color='gray',
                transform=axes[3].transAxes)
    
    # Confidence bar
    bar_width = 0.6
    bar_height = 0.05
    bar_x = 0.2
    bar_y = 0.05
    
    # Background bar
    axes[3].add_patch(plt.Rectangle((bar_x, bar_y), bar_width, bar_height, # type: ignore
                                    facecolor='lightgray', transform=axes[3].transAxes))
    
    # Probability bar
    axes[3].add_patch(plt.Rectangle((bar_x, bar_y), bar_width * probability, bar_height, # type: ignore
                                    facecolor=color, alpha=0.7, transform=axes[3].transAxes))
    
    # Bar labels
    axes[3].text(bar_x, bar_y - 0.03, '0%', ha='left', va='top',
                fontsize=10, transform=axes[3].transAxes)
    axes[3].text(bar_x + bar_width, bar_y - 0.03, '100%', ha='right', va='top',
                fontsize=10, transform=axes[3].transAxes)
    axes[3].text(bar_x + bar_width/2, bar_y - 0.03, '50%', ha='center', va='top',
                fontsize=10, transform=axes[3].transAxes)
    
    fig.suptitle(title, fontsize=14, weight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    plt.show()


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Transformer Model Inference Demo')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--mc_dropout', action='store_true', default=True,
                       help='Use MC Dropout for uncertainty')
    parser.add_argument('--n_passes', type=int, default=10,
                       help='Number of MC dropout passes')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to files')
    
    args = parser.parse_args()
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print("=" * 70)
    print("Transformer Inference Demo")
    print("=" * 70 + "\n")
    
    # Load model
    model = load_model(args.checkpoint, device=device)
    
    print("\n" + "=" * 70)
    print("Demo 1: Synthetic Earthquake")
    print("=" * 70 + "\n")
    
    # Create synthetic earthquake
    earthquake = create_synthetic_earthquake(duration=2.0, sampling_rate=100)
    
    # Preprocess
    earthquake_processed = preprocess_waveform(
        earthquake,
        sampling_rate=100,
        apply_bandpass=True,
        freqmin=1.0,
        freqmax=30.0,
        normalize=True
    )
    
    # Predict
    prob_eq, unc_eq = predict(model, earthquake_processed, device, 
                              args.mc_dropout, args.n_passes)
    
    print(f"Prediction: {'EARTHQUAKE' if prob_eq > 0.5 else 'NOISE'}")
    print(f"Probability: {prob_eq:.3f}")
    print(f"Uncertainty: {unc_eq:.3f}\n")
    
    # Plot
    save_path = 'demo_earthquake.png' if args.save_plots else None
    plot_waveform_and_prediction(earthquake_processed, prob_eq, unc_eq,
                                 "Demo 1: Synthetic Earthquake", save_path)
    
    print("\n" + "=" * 70)
    print("Demo 2: Synthetic Noise")
    print("=" * 70 + "\n")
    
    # Create synthetic noise
    noise = create_synthetic_noise(duration=2.0, sampling_rate=100)
    
    # Preprocess
    noise_processed = preprocess_waveform(
        noise,
        sampling_rate=100,
        apply_bandpass=True,
        freqmin=1.0,
        freqmax=30.0,
        normalize=True
    )
    
    # Predict
    prob_noise, unc_noise = predict(model, noise_processed, device,
                                    args.mc_dropout, args.n_passes)
    
    print(f"Prediction: {'EARTHQUAKE' if prob_noise > 0.5 else 'NOISE'}")
    print(f"Probability: {prob_noise:.3f}")
    print(f"Uncertainty: {unc_noise:.3f}\n")
    
    # Plot
    save_path = 'demo_noise.png' if args.save_plots else None
    plot_waveform_and_prediction(noise_processed, prob_noise, unc_noise,
                                 "Demo 2: Synthetic Noise", save_path)
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70 + "\n")
    
    print("Summary:")
    print(f"  Earthquake - Probability: {prob_eq:.3f}, Uncertainty: {unc_eq:.3f}")
    print(f"  Noise      - Probability: {prob_noise:.3f}, Uncertainty: {unc_noise:.3f}")
    print()


if __name__ == '__main__':
    main()
