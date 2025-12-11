"""
Real-time detection simulation module.

Simulates how the EEW_Transformer model would behave processing a long seismic
signal window-by-window (2-second windows) as would happen in a real EEW system.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RealTimeSimulator:
    """
    Simulates real-time detection processing of seismic signals.
    
    Divides a long waveform into 2-second sliding windows and processes
    them sequentially, tracking when the model first detects a P-wave.
    """
    
    def __init__(
        self,
        model,
        sampling_rate: float = 100.0,
        window_size: float = 2.0,
        detection_threshold: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: Trained Transformer model
            sampling_rate: Sampling rate in Hz
            window_size: Window size in seconds
            detection_threshold: Threshold for positive detection
            device: Device to run model on
        """
        self.model = model
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.window_samples = int(window_size * sampling_rate)
        self.detection_threshold = detection_threshold
        self.device = device
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def process_signal(
        self,
        waveform: np.ndarray,
        true_p_arrival_sample: Optional[int] = None,
        stride: Optional[int] = None,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Process a long waveform window-by-window in real-time.
        
        Args:
            waveform: Full waveform (3, n_samples) or (n_samples, 3)
            true_p_arrival_sample: True P-wave arrival for comparison
            stride: Stride between windows in samples. If None, uses window_size
            return_probabilities: If True, return all probabilities; else only detections
        
        Returns:
            Dictionary with:
                - 'window_predictions': List of window prediction dicts
                - 'first_detection_sample': First window start where model detected P
                - 'first_detection_time': First detection time in seconds
                - 'true_p_arrival_time': True P-wave time in seconds
                - 'detection_delay': Delay in seconds (negative if early)
                - 'detected': Boolean, was P-wave detected
        """
        # Ensure correct shape
        if waveform.shape[0] != 3:
            waveform = waveform.T
        
        n_samples = waveform.shape[1]
        
        # Default stride: non-overlapping windows
        if stride is None:
            stride = self.window_samples
        
        # Extract windows
        windows = []
        window_info = []
        
        for start_idx in range(0, n_samples - self.window_samples + 1, stride):
            end_idx = start_idx + self.window_samples
            window = waveform[:, start_idx:end_idx]
            windows.append(window)
            window_info.append({
                'start_sample': start_idx,
                'end_sample': end_idx,
            })
        
        logger.info(f"Processing {len(windows)} windows from waveform of {n_samples} samples")
        
        # Process windows through model
        window_predictions = []
        first_detection_sample = None
        
        with torch.no_grad():
            for window, info in zip(windows, window_info):
                # Prepare input: (3, 200) -> (1, 3, 200)
                input_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)
                
                # Get prediction
                logits = self.model(input_tensor)
                prob = torch.sigmoid(logits).item()
                
                is_detection = prob > self.detection_threshold
                
                pred_dict = {
                    'start_sample': info['start_sample'],
                    'end_sample': info['end_sample'],
                    'start_time': info['start_sample'] / self.sampling_rate,
                    'end_time': info['end_sample'] / self.sampling_rate,
                    'prediction': prob,
                    'is_detection': is_detection,
                }
                
                window_predictions.append(pred_dict)
                
                # Track first detection
                if is_detection and first_detection_sample is None:
                    first_detection_sample = info['start_sample']
                    logger.info(f"✓ First detection at window starting at sample {first_detection_sample} "
                              f"({first_detection_sample / self.sampling_rate:.2f}s)")
        
        # Prepare results
        results = {
            'window_predictions': window_predictions,
            'first_detection_sample': first_detection_sample,
            'first_detection_time': first_detection_sample / self.sampling_rate if first_detection_sample is not None else None,
            'detected': first_detection_sample is not None,
            'total_windows': len(windows),
        }
        
        # Add comparison with true P-wave if available
        if true_p_arrival_sample is not None:
            true_p_time = true_p_arrival_sample / self.sampling_rate
            results['true_p_arrival_sample'] = true_p_arrival_sample
            results['true_p_arrival_time'] = true_p_time
            
            if first_detection_sample is not None:
                detection_delay = (first_detection_sample - true_p_arrival_sample) / self.sampling_rate
                results['detection_delay'] = detection_delay
                results['detection_early'] = detection_delay < 0
                
                logger.info(f"True P-wave at: {true_p_time:.2f}s, "
                          f"Detected at: {results['first_detection_time']:.2f}s, "
                          f"Delay: {detection_delay:+.2f}s")
        
        return results
    
    def process_signal_with_uncertainty(
        self,
        waveform: np.ndarray,
        true_p_arrival_sample: Optional[int] = None,
        stride: Optional[int] = None,
        mc_passes: int = 10
    ) -> Dict:
        """
        Process signal with MC Dropout uncertainty estimation.
        
        Args:
            waveform: Full waveform (3, n_samples)
            true_p_arrival_sample: True P-wave arrival
            stride: Stride between windows
            mc_passes: Number of MC dropout passes
        
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        # Ensure correct shape
        if waveform.shape[0] != 3:
            waveform = waveform.T
        
        n_samples = waveform.shape[1]
        if stride is None:
            stride = self.window_samples
        
        # Extract windows
        windows = []
        window_info = []
        
        for start_idx in range(0, n_samples - self.window_samples + 1, stride):
            end_idx = start_idx + self.window_samples
            window = waveform[:, start_idx:end_idx]
            windows.append(window)
            window_info.append({
                'start_sample': start_idx,
                'end_sample': end_idx,
            })
        
        logger.info(f"Processing {len(windows)} windows with MC Dropout (passes={mc_passes})")
        
        # Process with uncertainty
        window_predictions = []
        first_detection_sample = None
        
        self.model.train()  # Enable dropout for MC
        
        with torch.no_grad():
            for window, info in zip(windows, window_info):
                # MC passes
                predictions = []
                for _ in range(mc_passes):
                    input_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)
                    logits = self.model(input_tensor)
                    prob = torch.sigmoid(logits).item()
                    predictions.append(prob)
                
                # Compute statistics
                mean_prob = np.mean(predictions)
                std_prob = np.std(predictions)
                
                is_detection = mean_prob > self.detection_threshold
                
                pred_dict = {
                    'start_sample': info['start_sample'],
                    'end_sample': info['end_sample'],
                    'start_time': info['start_sample'] / self.sampling_rate,
                    'end_time': info['end_sample'] / self.sampling_rate,
                    'mean_prediction': mean_prob,
                    'std_prediction': std_prob,
                    'is_detection': is_detection,
                }
                
                window_predictions.append(pred_dict)
                
                if is_detection and first_detection_sample is None:
                    first_detection_sample = info['start_sample']
        
        self.model.eval()  # Back to eval mode
        
        # Prepare results
        results = {
            'window_predictions': window_predictions,
            'first_detection_sample': first_detection_sample,
            'first_detection_time': first_detection_sample / self.sampling_rate if first_detection_sample is not None else None,
            'detected': first_detection_sample is not None,
            'total_windows': len(windows),
            'mc_passes': mc_passes,
        }
        
        if true_p_arrival_sample is not None:
            true_p_time = true_p_arrival_sample / self.sampling_rate
            results['true_p_arrival_sample'] = true_p_arrival_sample
            results['true_p_arrival_time'] = true_p_time
            
            if first_detection_sample is not None:
                detection_delay = (first_detection_sample - true_p_arrival_sample) / self.sampling_rate
                results['detection_delay'] = detection_delay
        
        return results


def simulate_realtime_detection(
    model,
    waveform: np.ndarray,
    sampling_rate: float = 100.0,
    window_size: float = 2.0,
    true_p_arrival_sample: Optional[int] = None,
    detection_threshold: float = 0.5,
    stride: Optional[int] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Convenience function to simulate real-time detection.
    
    Args:
        model: Trained Transformer model
        waveform: Full waveform (3, n_samples)
        sampling_rate: Sampling rate in Hz
        window_size: Window size in seconds
        true_p_arrival_sample: True P-wave arrival sample
        detection_threshold: Detection threshold
        stride: Stride between windows (default: window_size)
        device: Device to run on
    
    Returns:
        Results dictionary
    """
    simulator = RealTimeSimulator(
        model,
        sampling_rate=sampling_rate,
        window_size=window_size,
        detection_threshold=detection_threshold,
        device=device
    )
    
    return simulator.process_signal(
        waveform,
        true_p_arrival_sample=true_p_arrival_sample,
        stride=stride
    )


def get_detected_windows(
    results: Dict,
    min_probability: float = 0.5
) -> List[Dict]:
    """
    Extract detected windows from results.
    
    Args:
        results: Results dict from simulate_realtime_detection
        min_probability: Minimum probability threshold
    
    Returns:
        List of detected window dicts
    """
    detected = []
    for pred in results['window_predictions']:
        prob_key = 'prediction' if 'prediction' in pred else 'mean_prediction'
        if pred[prob_key] >= min_probability:
            detected.append(pred)
    
    return detected


def get_detection_statistics(
    results: Dict
) -> Dict:
    """
    Compute statistics from detection results.
    
    Args:
        results: Results dict from simulate_realtime_detection
    
    Returns:
        Statistics dictionary
    """
    predictions = results['window_predictions']
    
    # Extract probabilities
    prob_key = 'prediction' if 'prediction' in predictions[0] else 'mean_prediction'
    probs = [p[prob_key] for p in predictions]
    
    stats = {
        'total_windows': len(predictions),
        'mean_probability': np.mean(probs),
        'max_probability': np.max(probs),
        'min_probability': np.min(probs),
        'std_probability': np.std(probs),
        'num_detections': sum(1 for p in predictions if p[prob_key] > 0.5),
        'detection_rate': sum(1 for p in predictions if p[prob_key] > 0.5) / len(predictions),
    }
    
    # Add timing info if available
    if results.get('true_p_arrival_time') is not None:
        stats['true_p_arrival_time'] = results['true_p_arrival_time']
    
    if results.get('first_detection_time') is not None:
        stats['first_detection_time'] = results['first_detection_time']
        
        if results.get('detection_delay') is not None:
            stats['detection_delay'] = results['detection_delay']
    
    return stats
