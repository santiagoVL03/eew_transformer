"""
Lightweight baseline models for 2-second seismic window P-wave detection.

These baselines are specifically designed for 2-second windows (200 samples @ 100 Hz)
and use simple, fast features suitable for real-time EEW applications.

Baselines:
1. MLP: Simple neural network on engineered features
2. CNN 1D: Lightweight 1D convolutional network
3. Random Forest: Tree-based classifier on spectral features
4. SVM: Support Vector Machine on engineered features

All models operate on features extracted from 3-channel seismic signals.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.fft import fft


class FeatureExtractor:
    """Extract engineered features from 3-channel seismic signals."""
    
    def extract_features(self, waveform):
        """
        Extract features from 3-channel waveform (3, 200).
        
        Features extracted per channel:
        - RMS (Root Mean Square)
        - Peak amplitude
        - Energy
        - Spectral centroid
        - Spectral power in bands (1-5 Hz, 5-15 Hz, 15-30 Hz)
        - Zero-crossing rate
        - Envelope mean
        
        Args:
            waveform: np.array of shape (3, 200)
        
        Returns:
            features: np.array of shape (3 * n_features_per_channel,)
        """
        features = []
        
        for ch in range(waveform.shape[0]):
            trace = waveform[ch]
            
            # Time-domain features
            rms = np.sqrt(np.mean(trace ** 2))
            peak = np.max(np.abs(trace))
            energy = np.sum(trace ** 2)
            
            # Zero-crossing rate
            zcr = np.mean(np.abs(np.diff(np.sign(trace)))) / 2
            
            # Envelope
            analytic_signal = signal.hilbert(trace)
            envelope = np.abs(analytic_signal)
            envelope_mean = np.mean(envelope)
            
            # Spectral features
            freqs = np.fft.fftfreq(len(trace), d=1/100)  # 100 Hz sampling
            fft_mag = np.abs(fft(trace))
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs[:len(freqs)//2] * fft_mag[:len(freqs)//2]) / np.sum(fft_mag[:len(freqs)//2] + 1e-10)
            
            # Bandpower in standard EEW bands
            band_1_5_Hz = self._bandpower(trace, 1, 5, 100)
            band_5_15_Hz = self._bandpower(trace, 5, 15, 100)
            band_15_30_Hz = self._bandpower(trace, 15, 30, 100)
            
            # Stack features
            ch_features = [rms, peak, energy, zcr, envelope_mean, spectral_centroid,
                          band_1_5_Hz, band_5_15_Hz, band_15_30_Hz]
            features.extend(ch_features)
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def _bandpower(trace, low, high, fs, nperseg=100):
        """
        Compute power spectral density in a frequency band.
        
        Args:
            trace: 1D waveform
            low: Low frequency (Hz)
            high: High frequency (Hz)
            fs: Sampling frequency (Hz)
            nperseg: Length of each segment for Welch's method
        
        Returns:
            Bandpower (normalized)
        """
        freqs, psd = signal.welch(trace, fs=fs, nperseg=nperseg)
        mask = (freqs >= low) & (freqs <= high)
        bandpower = np.sum(psd[mask])
        
        return bandpower / (np.sum(psd) + 1e-10)
    
    def extract_batch(self, waveforms):
        """
        Extract features for batch of waveforms.
        
        Args:
            waveforms: np.array of shape (N, 3, 200)
        
        Returns:
            features: np.array of shape (N, n_features)
        """
        batch_features = []
        for wf in waveforms:
            feat = self.extract_features(wf)
            batch_features.append(feat)
        
        return np.array(batch_features, dtype=np.float32)


class MLPBaseline(nn.Module):
    """
    Lightweight MLP baseline for 2-second P-wave detection.
    
    Architecture:
    - Input: 27 features (9 features × 3 channels)
    - Hidden: 64 neurons (ReLU + Dropout)
    - Hidden: 32 neurons (ReLU + Dropout)
    - Output: 1 (sigmoid for binary classification)
    
    Parameters: ~3,000
    """
    
    def __init__(self, input_features=27, hidden1=64, hidden2=32, dropout=0.2):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        self.mlp = nn.Sequential(
            nn.Linear(input_features, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Waveforms of shape (batch, 3, 200) or already extracted features
        
        Returns:
            logits of shape (batch, 1)
        """
        if x.dim() == 3:
            # Extract features if input is raw waveforms
            x = x.cpu().numpy()
            features = self.feature_extractor.extract_batch(x)
            x = torch.from_numpy(features).to(x.device if isinstance(x, torch.Tensor) else 'cpu')
        
        return self.mlp(x)
    
    def predict(self, X):
        """
        Predict labels for input features.
        
        Args:
            X: Features of shape (N, 27) as numpy array
        
        Returns:
            predictions of shape (N,) as binary labels (0 or 1)
        """
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            logits = self.forward(X)
            predictions = (torch.sigmoid(logits) > 0.5).long().squeeze(-1).numpy()
        return predictions
    
    def predict_proba(self, X):
        """
        Predict probabilities for input features.
        
        Args:
            X: Features of shape (N, 27) as numpy array
        
        Returns:
            probabilities of shape (N, 2) as [prob_class_0, prob_class_1]
        """
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            logits = self.forward(X)
            probs = torch.sigmoid(logits).squeeze(-1).numpy()
        return np.column_stack([1 - probs, probs])
    
    def fit(self, X, y, epochs=10, batch_size=32, lr=0.001):
        """
        Fit MLP model on data.
        
        Args:
            X: Waveforms of shape (N, 3, 200)
            y: Labels of shape (N,)
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
        """
        # Extract features if needed
        if len(X.shape) == 3:
            features = self.feature_extractor.extract_batch(X)
        else:
            features = X
        
        # Standardize features
        features = self.scaler.fit_transform(features)
        
        # Prepare data
        X_tensor = torch.from_numpy(features).float()
        y_tensor = torch.from_numpy(y).float().unsqueeze(-1)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        self.train()
        for epoch in range(epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.forward(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
        
        self.is_fitted = True


class CNN1DBaseline(nn.Module):
    """
    Lightweight 1D CNN baseline for 2-second P-wave detection.
    
    Architecture:
    - Input: (batch, 3, 200)
    - Conv1D: 16 filters, kernel=7
    - Conv1D: 32 filters, kernel=5
    - Global average pooling
    - Dense: 64 neurons (ReLU)
    - Dense: 1 (sigmoid)
    
    Parameters: ~12,000
    """
    
    def __init__(self, input_channels=3, dropout=0.2):
        super().__init__()
        
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2),
            
            # Block 2
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            
            # Block 3
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Waveforms of shape (batch, 3, 200)
        
        Returns:
            logits of shape (batch, 1)
        """
        x = self.cnn(x)
        x = x.squeeze(-1)  # Remove last dimension from pooling
        x = self.classifier(x)
        return x
    
    def predict(self, X):
        """
        Predict labels for input waveforms.
        
        Args:
            X: Waveforms of shape (N, 3, 200) as numpy array or torch tensor
        
        Returns:
            predictions of shape (N,) as binary labels (0 or 1)
        """
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            logits = self.forward(X)
            probs = torch.sigmoid(logits).squeeze(-1)
            predictions = (probs > 0.5).long().numpy()
        return predictions
    
    def predict_proba(self, X):
        """
        Predict probabilities for input waveforms.
        
        Args:
            X: Waveforms of shape (N, 3, 200) as numpy array or torch tensor
        
        Returns:
            probabilities of shape (N, 2) as [prob_class_0, prob_class_1]
        """
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            logits = self.forward(X)
            probs = torch.sigmoid(logits).squeeze(-1).numpy()
        return np.column_stack([1 - probs, probs])


class RandomForestBaseline:
    """
    Random Forest baseline using engineered features.
    
    Parameters: ~20,000 (stored tree structure)
    """
    
    def __init__(self, n_estimators=100, max_depth=15, random_state=42):
        self.feature_extractor = FeatureExtractor()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit Random Forest model.
        
        Args:
            X: Waveforms of shape (N, 3, 200)
            y: Labels of shape (N,)
        """
        features = self.feature_extractor.extract_batch(X)
        features = self.scaler.fit_transform(features)
        self.model.fit(features, y)
        self.is_fitted = True
    
    def predict(self, X):
        """
        Predict labels.
        
        Args:
            X: Waveforms of shape (N, 3, 200) or numpy array of features
        
        Returns:
            predictions of shape (N,)
        """
        if len(X.shape) == 3:
            features = self.feature_extractor.extract_batch(X)
        else:
            features = X
        
        features = self.scaler.transform(features)
        return self.model.predict(features)
    
    def predict_proba(self, X):
        """
        Predict probability.
        
        Args:
            X: Waveforms of shape (N, 3, 200)
        
        Returns:
            probabilities of shape (N, 2) or (N,)
        """
        if len(X.shape) == 3:
            features = self.feature_extractor.extract_batch(X)
        else:
            features = X
        
        features = self.scaler.transform(features)
        return self.model.predict_proba(features)


class SVMBaseline:
    """
    SVM baseline using engineered features.
    
    Parameters: ~500-2000 (support vectors)
    """
    
    def __init__(self, kernel='rbf', C=1.0, random_state=42):
        self.feature_extractor = FeatureExtractor()
        self.model = SVC(
            kernel=kernel,
            C=C,
            probability=True,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit SVM model.
        
        Args:
            X: Waveforms of shape (N, 3, 200)
            y: Labels of shape (N,)
        """
        features = self.feature_extractor.extract_batch(X)
        features = self.scaler.fit_transform(features)
        self.model.fit(features, y)
        self.is_fitted = True
    
    def predict(self, X):
        """
        Predict labels.
        
        Args:
            X: Waveforms of shape (N, 3, 200)
        
        Returns:
            predictions of shape (N,)
        """
        if len(X.shape) == 3:
            features = self.feature_extractor.extract_batch(X)
        else:
            features = X
        
        features = self.scaler.transform(features)
        return self.model.predict(features)
    
    def predict_proba(self, X):
        """
        Predict probability.
        
        Args:
            X: Waveforms of shape (N, 3, 200)
        
        Returns:
            probabilities of shape (N, 2)
        """
        if len(X.shape) == 3:
            features = self.feature_extractor.extract_batch(X)
        else:
            features = X
        
        features = self.scaler.transform(features)
        return self.model.predict_proba(features)


def create_baseline(baseline_type='mlp', device='cpu', **kwargs):
    """
    Factory function to create baseline model.
    
    Args:
        baseline_type: 'mlp', 'cnn1d', 'random_forest', 'svm'
        device: Device for neural network models
        **kwargs: Additional arguments for model
    
    Returns:
        Baseline model
    """
    if baseline_type == 'mlp':
        return MLPBaseline(**kwargs)
    elif baseline_type == 'cnn1d':
        return CNN1DBaseline(**kwargs)
    elif baseline_type == 'random_forest':
        return RandomForestBaseline(**kwargs)
    elif baseline_type == 'svm':
        return SVMBaseline(**kwargs)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
