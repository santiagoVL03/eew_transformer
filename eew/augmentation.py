"""
Data augmentation for seismic waveforms.

Includes:
- Gaussian noise addition
- Amplitude scaling
- Time shifting
- SNR-based noise injection
"""

import numpy as np


class WaveformAugmenter:
    """
    Data augmentation for seismic waveforms.
    """
    
    def __init__(
        self,
        add_noise=True,
        noise_snr_range=(5, 20),
        scale_amplitude=True,
        scale_range=(0.8, 1.2),
        time_shift=True,
        shift_range=(-0.1, 0.1),
        sampling_rate=100,
        p=0.5
    ):
        """
        Args:
            add_noise: Whether to add Gaussian noise
            noise_snr_range: SNR range in dB (min, max)
            scale_amplitude: Whether to scale amplitude
            scale_range: Amplitude scaling range (min, max)
            time_shift: Whether to apply time shifting
            shift_range: Time shift range in seconds (min, max)
            sampling_rate: Sampling rate in Hz
            p: Probability of applying each augmentation
        """
        self.add_noise = add_noise
        self.noise_snr_range = noise_snr_range
        self.scale_amplitude = scale_amplitude
        self.scale_range = scale_range
        self.time_shift = time_shift
        self.shift_range = shift_range
        self.sampling_rate = sampling_rate
        self.p = p
    
    def __call__(self, waveform):
        """
        Apply augmentation to waveform.
        
        Args:
            waveform: Input waveform (channels, samples) as numpy array
        
        Returns:
            Augmented waveform
        """
        augmented = waveform.copy()
        
        # Add noise
        if self.add_noise and np.random.rand() < self.p:
            augmented = self._add_noise(augmented)
        
        # Scale amplitude
        if self.scale_amplitude and np.random.rand() < self.p:
            augmented = self._scale_amplitude(augmented)
        
        # Time shift
        if self.time_shift and np.random.rand() < self.p:
            augmented = self._time_shift(augmented)
        
        return augmented
    
    def _add_noise(self, waveform):
        """
        Add Gaussian noise at random SNR.
        
        SNR (dB) = 10 * log10(signal_power / noise_power)
        """
        # Random SNR in range
        snr_db = np.random.uniform(*self.noise_snr_range)
        
        noisy = np.zeros_like(waveform)
        
        for i in range(waveform.shape[0]):
            signal = waveform[i]
            
            # Calculate signal power
            signal_power = np.mean(signal ** 2)
            
            # Calculate required noise power from SNR
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            
            # Generate noise with target power
            noise = np.random.randn(len(signal))
            noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))
            
            noisy[i] = signal + noise
        
        return noisy
    
    def _scale_amplitude(self, waveform):
        """
        Scale waveform amplitude by random factor.
        """
        scale = np.random.uniform(*self.scale_range)
        return waveform * scale
    
    def _time_shift(self, waveform):
        """
        Shift waveform in time by random amount.
        """
        # Random shift in seconds
        shift_sec = np.random.uniform(*self.shift_range)
        shift_samples = int(shift_sec * self.sampling_rate)
        
        if shift_samples == 0:
            return waveform
        
        shifted = np.zeros_like(waveform)
        
        if shift_samples > 0:
            # Shift right (delay)
            shifted[:, shift_samples:] = waveform[:, :-shift_samples]
        else:
            # Shift left (advance)
            shifted[:, :shift_samples] = waveform[:, -shift_samples:]
        
        return shifted


class NoAugmentation:
    """
    Dummy augmenter that does nothing (for validation/test sets).
    """
    
    def __call__(self, waveform):
        return waveform


if __name__ == '__main__':
    # Test augmentation
    print("Testing waveform augmentation...")
    
    # Create synthetic waveform
    sr = 100
    duration = 2
    t = np.linspace(0, duration, sr * duration)
    
    waveform = np.zeros((3, len(t)))
    waveform[0] = np.sin(2 * np.pi * 5 * t)
    waveform[1] = np.sin(2 * np.pi * 7 * t)
    waveform[2] = np.sin(2 * np.pi * 10 * t)
    
    print(f"Original shape: {waveform.shape}")
    print(f"Original range: [{waveform.min():.3f}, {waveform.max():.3f}]")
    
    # Create augmenter
    augmenter = WaveformAugmenter(
        add_noise=True,
        scale_amplitude=True,
        time_shift=True,
        p=1.0  # Always apply for testing
    )
    
    # Apply augmentation
    augmented = augmenter(waveform)
    
    print(f"Augmented shape: {augmented.shape}")
    print(f"Augmented range: [{augmented.min():.3f}, {augmented.max():.3f}]")
    
    # Test multiple times
    print("\nApplying 5 random augmentations:")
    augmenter.p = 0.5
    for i in range(5):
        aug = augmenter(waveform)
        print(f"  {i+1}. Range: [{aug.min():.3f}, {aug.max():.3f}]")
    
    print("\nTest passed!")
