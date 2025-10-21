"""
Waveform preprocessing utilities for Transformer.

Includes:
- Bandpass filtering (1-30 Hz)
- Resampling
- Normalization
- Phase-centered window extraction
"""

import numpy as np
from scipy import signal
from obspy.signal.filter import bandpass


def bandpass_filter(waveform, freqmin=1.0, freqmax=30.0, sampling_rate=100, corners=4):
    """
    Apply bandpass filter to waveform using ObsPy.
    
    Args:
        waveform: Input waveform (channels, samples)
        freqmin: Minimum frequency (Hz)
        freqmax: Maximum frequency (Hz)
        sampling_rate: Sampling rate (Hz)
        corners: Number of corners for Butterworth filter
    
    Returns:
        Filtered waveform
    """
    filtered = np.zeros_like(waveform)
    
    for i in range(waveform.shape[0]):
        try:
            filtered[i] = bandpass(
                waveform[i],
                freqmin=freqmin,
                freqmax=freqmax,
                df=sampling_rate,
                corners=corners,
                zerophase=True
            )
        except Exception as e:
            # If ObsPy fails, use scipy
            nyquist = sampling_rate / 2
            low = freqmin / nyquist
            high = freqmax / nyquist
            sos = signal.butter(corners, [low, high], btype='band', output='sos')
            filtered[i] = signal.sosfilt(sos, waveform[i])
    
    return filtered


def resample_waveform(waveform, original_sr, target_sr):
    """
    Resample waveform to target sampling rate.
    
    Args:
        waveform: Input waveform (channels, samples)
        original_sr: Original sampling rate (Hz)
        target_sr: Target sampling rate (Hz)
    
    Returns:
        Resampled waveform
    """
    if original_sr == target_sr:
        return waveform
    
    num_samples = int(waveform.shape[1] * target_sr / original_sr)
    resampled = signal.resample(waveform, num_samples, axis=1)
    
    return resampled


def normalize_waveform(waveform, method='minmax', eps=1e-10):
    """
    Normalize waveform.
    
    Args:
        waveform: Input waveform (channels, samples)
        method: Normalization method ('minmax', 'zscore', 'std')
        eps: Small value to avoid division by zero
    
    Returns:
        Normalized waveform
    """
    normalized = np.zeros_like(waveform)
    
    for i in range(waveform.shape[0]):
        trace = waveform[i]
        
        if method == 'minmax':
            # Min-max normalization to [0, 1] or [-1, 1]
            min_val = trace.min()
            max_val = trace.max()
            if max_val - min_val > eps:
                normalized[i] = (trace - min_val) / (max_val - min_val + eps)
            else:
                normalized[i] = trace
        
        elif method == 'zscore':
            # Z-score normalization (zero mean, unit variance)
            mean = trace.mean()
            std = trace.std()
            if std > eps:
                normalized[i] = (trace - mean) / (std + eps)
            else:
                normalized[i] = trace - mean
        
        elif method == 'std':
            # Standard deviation normalization (preserve mean)
            std = trace.std()
            if std > eps:
                normalized[i] = trace / (std + eps)
            else:
                normalized[i] = trace
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def extract_phase_window(waveform, pick_sample, window_before=1.0, window_after=1.0, 
                         sampling_rate=100, pad_mode='constant'):
    """
    Extract window centered around phase pick.
    
    Args:
        waveform: Input waveform (channels, samples)
        pick_sample: Sample index of phase pick
        window_before: Seconds before pick (default 1.0)
        window_after: Seconds after pick (default 1.0)
        sampling_rate: Sampling rate (Hz)
        pad_mode: Padding mode if window extends beyond trace
    
    Returns:
        Windowed waveform (channels, window_samples)
    """
    samples_before = int(window_before * sampling_rate)
    samples_after = int(window_after * sampling_rate)
    
    start_idx = pick_sample - samples_before
    end_idx = pick_sample + samples_after
    
    window_size = samples_before + samples_after
    
    # Handle edge cases with padding
    if start_idx < 0 or end_idx > waveform.shape[1]:
        windowed = np.zeros((waveform.shape[0], window_size))
        
        # Calculate valid slice
        valid_start = max(0, start_idx)
        valid_end = min(waveform.shape[1], end_idx)
        
        # Calculate where to place valid data in window
        window_start = max(0, -start_idx)
        window_end = window_start + (valid_end - valid_start)
        
        windowed[:, window_start:window_end] = waveform[:, valid_start:valid_end]
    else:
        windowed = waveform[:, start_idx:end_idx]
    
    return windowed


def preprocess_waveform(
    waveform,
    sampling_rate=100,
    target_sr=100,
    apply_bandpass=True,
    freqmin=1.0,
    freqmax=30.0,
    normalize=True,
    norm_method='minmax',
    pick_sample=None,
    window_before=1.0,
    window_after=1.0
):
    """
    Complete preprocessing pipeline for a single waveform.
    
    Args:
        waveform: Input waveform (channels, samples)
        sampling_rate: Original sampling rate (Hz)
        target_sr: Target sampling rate (Hz)
        apply_bandpass: Whether to apply bandpass filter
        freqmin: Bandpass minimum frequency (Hz)
        freqmax: Bandpass maximum frequency (Hz)
        normalize: Whether to normalize
        norm_method: Normalization method
        pick_sample: Phase pick sample (if None, use center)
        window_before: Seconds before pick
        window_after: Seconds after pick
    
    Returns:
        Preprocessed waveform
    """
    processed = waveform.copy()
    
    # 1. Bandpass filter
    if apply_bandpass:
        processed = bandpass_filter(
            processed,
            freqmin=freqmin,
            freqmax=freqmax,
            sampling_rate=sampling_rate
        )
    
    # 2. Resample
    if sampling_rate != target_sr:
        processed = resample_waveform(processed, sampling_rate, target_sr)
        # Update pick sample after resampling
        if pick_sample is not None:
            pick_sample = int(pick_sample * target_sr / sampling_rate)
        sampling_rate = target_sr
    
    # 3. Extract window
    if pick_sample is not None:
        processed = extract_phase_window(
            processed,
            pick_sample,
            window_before=window_before,
            window_after=window_after,
            sampling_rate=sampling_rate
        )
    
    # 4. Normalize
    if normalize:
        processed = normalize_waveform(processed, method=norm_method)
    
    return processed


if __name__ == '__main__':
    # Test preprocessing
    print("Testing preprocessing...")
    
    # Create synthetic waveform
    sr = 100
    duration = 5  # seconds
    t = np.linspace(0, duration, sr * duration)
    
    # 3-component synthetic signal
    waveform = np.zeros((3, len(t)))
    waveform[0] = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(len(t))
    waveform[1] = np.sin(2 * np.pi * 7 * t + 0.5) + 0.5 * np.random.randn(len(t))
    waveform[2] = np.sin(2 * np.pi * 10 * t + 1.0) + 0.5 * np.random.randn(len(t))
    
    print(f"Input shape: {waveform.shape}")
    
    # Preprocess
    pick_sample = len(t) // 2  # Pick at center
    processed = preprocess_waveform(
        waveform,
        sampling_rate=sr,
        pick_sample=pick_sample,
        window_before=1.0,
        window_after=1.0
    )
    
    print(f"Output shape: {processed.shape}")
    print(f"Output range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    print("\nTest passed!")
