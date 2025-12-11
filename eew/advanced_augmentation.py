"""
Advanced data augmentation techniques for seismic waveforms.

Includes:
- Mixup: Linear interpolation between samples
- Channel dropout: Randomly drop channels
- Spectral augmentation
"""

import numpy as np
import torch


class MixupAugmenter:
    """
    Mixup data augmentation for waveforms.
    
    Generates virtual training examples by mixing pairs of samples:
        x_mixed = lambda * x_i + (1 - lambda) * x_j
        y_mixed = lambda * y_i + (1 - lambda) * y_j
    
    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2017)
    """
    
    def __init__(self, alpha=0.2, p=0.5):
        """
        Args:
            alpha: Beta distribution parameter (alpha > 0)
                   Typical values: 0.1-0.4 (higher = more mixing)
            p: Probability of applying mixup
        """
        self.alpha = alpha
        self.p = p
    
    def __call__(self, waveforms, labels):
        """
        Apply mixup augmentation to a batch.
        
        Args:
            waveforms: Batch of waveforms (batch_size, channels, samples)
            labels: Batch of labels (batch_size, 1)
        
        Returns:
            mixed_waveforms, mixed_labels
        """
        if np.random.rand() > self.p:
            return waveforms, labels
        
        batch_size = waveforms.size(0)
        
        # Sample mixing coefficient from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Random permutation for pairing
        index = torch.randperm(batch_size).to(waveforms.device)
        
        # Mix waveforms and labels
        mixed_waveforms = lam * waveforms + (1 - lam) * waveforms[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_waveforms, mixed_labels


class ChannelDropout:
    """
    Randomly drop one or more channels during training.
    
    Forces the model to not rely on a single channel.
    """
    
    def __init__(self, drop_prob=0.1, p=0.3):
        """
        Args:
            drop_prob: Probability of dropping each channel
            p: Probability of applying channel dropout
        """
        self.drop_prob = drop_prob
        self.p = p
    
    def __call__(self, waveform):
        """
        Apply channel dropout.
        
        Args:
            waveform: Waveform (channels, samples) as numpy array or torch tensor
        
        Returns:
            Waveform with channels potentially dropped
        """
        if np.random.rand() > self.p:
            return waveform
        
        is_torch = isinstance(waveform, torch.Tensor)
        if is_torch:
            device = waveform.device
            waveform = waveform.cpu().numpy()
        
        result = waveform.copy()
        
        # Randomly drop channels (at least keep 1 channel)
        n_channels = waveform.shape[0]
        for i in range(n_channels):
            if np.random.rand() < self.drop_prob and np.any(result != 0):
                result[i] = 0
        
        # Ensure at least one channel is active
        if np.all(result == 0):
            random_channel = np.random.randint(n_channels)
            result[random_channel] = waveform[random_channel]
        
        if is_torch:
            result = torch.from_numpy(result).to(device)
        
        return result


class AdvancedWaveformAugmenter:
    """
    Advanced waveform augmentation combining multiple techniques.
    """
    
    # Contador de estadísticas (compartido entre todas las instancias)
    _stats = {
        'total_processed': 0,
        'augmentations_applied': 0
    }
    
    def __init__(
        self,
        # Original augmentations
        add_noise=True,
        noise_snr_range=(5, 20),
        scale_amplitude=True,
        scale_range=(0.8, 1.2),
        time_shift=True,
        shift_range=(-0.1, 0.1),
        # New augmentations
        channel_dropout=True,
        channel_drop_prob=0.1,
        baseline_drift=True,
        drift_range=(-0.1, 0.1),
        # General
        sampling_rate=100,
        p=0.5,
        track_stats=True
    ):
        """
        Args:
            add_noise: Whether to add Gaussian noise
            noise_snr_range: SNR range in dB (min, max)
            scale_amplitude: Whether to scale amplitude
            scale_range: Amplitude scaling range (min, max)
            time_shift: Whether to apply time shifting
            shift_range: Time shift range in seconds (min, max)
            channel_dropout: Whether to apply channel dropout
            channel_drop_prob: Probability of dropping each channel
            baseline_drift: Whether to add baseline drift
            drift_range: Drift range (min, max)
            sampling_rate: Sampling rate in Hz
            p: Base probability of applying each augmentation
            track_stats: Whether to track augmentation statistics
        """
        self.add_noise = add_noise
        self.noise_snr_range = noise_snr_range
        self.scale_amplitude = scale_amplitude
        self.scale_range = scale_range
        self.time_shift = time_shift
        self.shift_range = shift_range
        self.channel_dropout = channel_dropout
        self.channel_drop_prob = channel_drop_prob
        self.baseline_drift = baseline_drift
        self.drift_range = drift_range
        self.sampling_rate = sampling_rate
        self.p = p
        self.track_stats = track_stats
        
        # Initialize sub-augmenters
        self.channel_dropout_augmenter = ChannelDropout(
            drop_prob=channel_drop_prob, p=p
        )
    
    def __call__(self, waveform):
        """
        Apply augmentation to waveform.
        
        Args:
            waveform: Input waveform (channels, samples) as numpy array
        
        Returns:
            Augmented waveform
        """
        if self.track_stats:
            AdvancedWaveformAugmenter._stats['total_processed'] += 1
        
        augmented = waveform.copy()
        aug_count = 0
        
        # Add noise
        if self.add_noise and np.random.rand() < self.p:
            augmented = self._add_noise(augmented)
            aug_count += 1
        
        # Scale amplitude
        if self.scale_amplitude and np.random.rand() < self.p:
            augmented = self._scale_amplitude(augmented)
            aug_count += 1
        
        # Time shift
        if self.time_shift and np.random.rand() < self.p:
            augmented = self._time_shift(augmented)
            aug_count += 1
        
        # Channel dropout
        if self.channel_dropout and np.random.rand() < self.p:
            augmented = self.channel_dropout_augmenter(augmented)
            aug_count += 1
        
        # Baseline drift
        if self.baseline_drift and np.random.rand() < self.p:
            augmented = self._add_baseline_drift(augmented)
            aug_count += 1
        
        if self.track_stats and aug_count > 0:
            AdvancedWaveformAugmenter._stats['augmentations_applied'] += aug_count
        
        return augmented
    
    @classmethod
    def print_stats(cls):
        """Imprime las estadísticas de augmentación."""
        stats = cls._stats
        total = stats['total_processed']
        
        if total == 0:
            print("\n⚠️  No se han procesado señales con AdvancedWaveformAugmenter aún")
            return
        
        print("\n" + "="*60)
        print("ESTADÍSTICAS DE AUGMENTACIÓN - AdvancedWaveformAugmenter")
        print("="*60)
        print(f"Total de señales procesadas: {total:,}")
        print(f"Augmentaciones aplicadas:    {stats['augmentations_applied']:,}")
        print(f"Promedio aug/señal:          {stats['augmentations_applied']/total:.2f}")
        print("  (noise, scale, time_shift, channel_dropout, baseline_drift)")
        print("="*60 + "\n")
    
    @classmethod
    def reset_stats(cls):
        """Reinicia las estadísticas."""
        cls._stats = {
            'total_processed': 0,
            'augmentations_applied': 0
        }
    
    def _add_noise(self, waveform):
        """Add Gaussian noise at random SNR."""
        snr_db = np.random.uniform(*self.noise_snr_range)
        
        noisy = np.zeros_like(waveform)
        
        for i in range(waveform.shape[0]):
            signal = waveform[i]
            signal_power = np.mean(signal ** 2)
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            
            noise = np.random.randn(len(signal))
            noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))
            
            noisy[i] = signal + noise
        
        return noisy
    
    def _scale_amplitude(self, waveform):
        """Scale amplitude randomly."""
        scale = np.random.uniform(*self.scale_range)
        return waveform * scale
    
    def _time_shift(self, waveform):
        """Shift waveform in time."""
        shift_seconds = np.random.uniform(*self.shift_range)
        shift_samples = int(shift_seconds * self.sampling_rate)
        
        if shift_samples == 0:
            return waveform
        
        shifted = np.zeros_like(waveform)
        
        if shift_samples > 0:
            shifted[:, shift_samples:] = waveform[:, :-shift_samples]
        else:
            shifted[:, :shift_samples] = waveform[:, -shift_samples:]
        
        return shifted
    
    def _add_baseline_drift(self, waveform):
        """Add slow baseline drift to simulate instrument drift."""
        drift = np.zeros_like(waveform)
        
        for i in range(waveform.shape[0]):
            # Random linear drift
            drift_amount = np.random.uniform(*self.drift_range)
            drift[i] = np.linspace(0, drift_amount, waveform.shape[1])
        
        return waveform + drift
