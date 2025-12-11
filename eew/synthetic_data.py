"""
Synthetic seismic data generation for 2-second windows.

Generates realistic synthetic P-wave and noise signals for testing
EEW models when STEAD data is not available.
"""

import numpy as np
from typing import Optional
from scipy import signal


def generate_synthetic_earthquake(
    duration: float = 2.0,
    sampling_rate: float = 100.0,
    freq_min: float = 1.0,
    freq_max: float = 15.0,
    amplitude: float = 1.0,
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Generate synthetic P-wave earthquake signal (3-channel).
    
    Args:
        duration: Duration in seconds
        sampling_rate: Sampling rate in Hz
        freq_min: Minimum frequency (Hz)
        freq_max: Maximum frequency (Hz)
        amplitude: Signal amplitude
        noise_level: Gaussian noise level
    
    Returns:
        Signal of shape (3, n_samples)
    """
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Create bandlimited signal (1-15 Hz for P-waves)
    signal_base = np.zeros((3, n_samples))
    
    # Add multiple frequency components
    freqs = np.linspace(freq_min, freq_max, 5)
    for ch in range(3):
        for freq in freqs:
            # Variable amplitude envelope (box-car for P-wave)
            envelope = np.ones(n_samples)
            # Taper at edges
            taper_len = int(0.1 * n_samples)
            envelope[:taper_len] = np.linspace(0, 1, taper_len)
            envelope[-taper_len:] = np.linspace(1, 0, taper_len)
            
            # Generate signal
            phase = np.random.uniform(0, 2*np.pi)
            component = np.sin(2*np.pi*freq*t + phase) * envelope
            signal_base[ch] += component / len(freqs)
    
    # Normalize
    signal_base = signal_base / np.max(np.abs(signal_base))
    
    # Add noise
    noise = np.random.randn(3, n_samples) * noise_level
    signal_base = signal_base * amplitude + noise
    
    return signal_base.astype(np.float32)


def generate_synthetic_noise(
    duration: float = 2.0,
    sampling_rate: float = 100.0,
    amplitude: float = 0.1,
    freq_max: float = 30.0
) -> np.ndarray:
    """
    Generate synthetic ambient noise signal (3-channel).
    
    Args:
        duration: Duration in seconds
        sampling_rate: Sampling rate in Hz
        amplitude: Noise amplitude
        freq_max: Maximum frequency (Hz)
    
    Returns:
        Noise signal of shape (3, n_samples)
    """
    n_samples = int(duration * sampling_rate)
    
    # White noise
    noise = np.random.randn(3, n_samples) * amplitude
    
    # Low-pass filter to create more realistic 1/f noise
    sos = signal.butter(4, freq_max, fs=sampling_rate, output='sos')
    for ch in range(3):
        noise[ch] = signal.sosfilt(sos, noise[ch])
    
    return noise.astype(np.float32)


def generate_synthetic_dataset(
    n_earthquakes: int = 1000,
    n_noise: int = 1000,
    duration: float = 2.0,
    sampling_rate: float = 100.0,
    seed: Optional[int] = None
) -> tuple:
    """
    Generate synthetic balanced dataset.
    
    Args:
        n_earthquakes: Number of earthquake samples
        n_noise: Number of noise samples
        duration: Duration in seconds
        sampling_rate: Sampling rate in Hz
        seed: Random seed for reproducibility
    
    Returns:
        (X, y) where X is (n_samples, 3, n_timepoints) and y is (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = []
    y = []
    
    # Generate earthquakes
    for i in range(n_earthquakes):
        amp = np.random.uniform(0.5, 2.0)
        freq_min = np.random.uniform(0.5, 5.0)
        freq_max = np.random.uniform(10.0, 25.0)
        noise_level = np.random.uniform(0.05, 0.2)
        
        sig = generate_synthetic_earthquake(
            duration=duration,
            sampling_rate=sampling_rate,
            freq_min=freq_min,
            freq_max=freq_max,
            amplitude=amp,
            noise_level=noise_level
        )
        X.append(sig)
        y.append(1)
    
    # Generate noise
    for i in range(n_noise):
        amp = np.random.uniform(0.05, 0.2)
        
        noise = generate_synthetic_noise(
            duration=duration,
            sampling_rate=sampling_rate,
            amplitude=amp,
            freq_max=30.0
        )
        X.append(noise)
        y.append(0)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y
