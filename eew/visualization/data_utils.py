"""
Data utility functions for loading and preparing STEAD samples for visualization.

Integrates with the existing data_loader.py to retrieve samples efficiently.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import torch
import logging

logger = logging.getLogger(__name__)


def load_stead_sample(
    stead_dataset,
    sample_idx: int,
    sampling_rate: float = 100.0,
    window_size: float = 2.0
) -> Dict:
    """
    Load a single STEAD sample with metadata.
    
    Args:
        stead_dataset: STEADDataset instance or seisbench dataset
        sample_idx: Index of sample to load
        sampling_rate: Sampling rate in Hz
        window_size: Window size in seconds
    
    Returns:
        Dictionary with:
            - 'waveform': (3, n_samples) array
            - 'metadata': Sample metadata
            - 'p_arrival_sample': P-wave arrival sample
            - 'label': 0 (noise) or 1 (event)
    """
    try:
        # Try with STEADDataset
        if hasattr(stead_dataset, 'waveforms'):
            # Preloaded mode
            waveform = stead_dataset.waveforms[sample_idx]
            label = stead_dataset.labels[sample_idx]
            
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.numpy()
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            p_arrival = None
            if stead_dataset.phase_picks is not None:
                p_arrival = stead_dataset.phase_picks[sample_idx]
            
            metadata = {}
            if stead_dataset.metadata is not None:
                metadata = stead_dataset.metadata.get(sample_idx, {})
            
            return {
                'waveform': waveform,
                'label': int(label),
                'p_arrival_sample': p_arrival,
                'metadata': metadata,
            }
        
        # Try with seisbench dataset
        elif hasattr(stead_dataset, 'get_sample'):
            waveform, metadata = stead_dataset.get_sample(sample_idx)
            
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.numpy()
            
            # Ensure 3 channels
            if waveform.shape[0] != 3:
                if waveform.shape[0] < 3:
                    waveform = np.pad(waveform, ((0, 3 - waveform.shape[0]), (0, 0)))
                else:
                    waveform = waveform[:3, :]
            
            # Determine label
            label = 1 if 'trace_p_arrival_sample' in metadata else 0
            p_arrival = metadata.get('trace_p_arrival_sample', None)
            
            return {
                'waveform': waveform,
                'label': label,
                'p_arrival_sample': p_arrival,
                'metadata': metadata,
            }
    
    except Exception as e:
        logger.error(f"Error loading sample {sample_idx}: {e}")
        return None


def extract_window(
    waveform: np.ndarray,
    start_sample: int,
    window_samples: int = 200,
    p_arrival_sample: Optional[int] = None
) -> Tuple[np.ndarray, Optional[int]]:
    """
    Extract a window from a waveform.
    
    Args:
        waveform: Full waveform (3, n_samples)
        start_sample: Starting sample index
        window_samples: Number of samples in window
        p_arrival_sample: Absolute P-wave arrival sample
    
    Returns:
        Tuple of (window, p_arrival_in_window)
            where p_arrival_in_window is relative to window start
    """
    end_sample = min(start_sample + window_samples, waveform.shape[1])
    window = waveform[:, start_sample:end_sample]
    
    # Pad if necessary
    if window.shape[1] < window_samples:
        padding = ((0, 0), (0, window_samples - window.shape[1]))
        window = np.pad(window, padding, mode='constant')
    
    # Compute P-arrival relative to window
    p_arrival_in_window = None
    if p_arrival_sample is not None:
        p_arrival_in_window = p_arrival_sample - start_sample
        if p_arrival_in_window < 0 or p_arrival_in_window >= window_samples:
            p_arrival_in_window = None
    
    return window, p_arrival_in_window


def load_random_samples(
    stead_dataset,
    n_samples: int = 9,
    seed: Optional[int] = None
) -> List[Dict]:
    """
    Load random samples from STEAD dataset.
    
    Args:
        stead_dataset: STEADDataset instance
        n_samples: Number of samples to load
        seed: Random seed
    
    Returns:
        List of sample dictionaries
    """
    if seed is not None:
        np.random.seed(seed)
    
    dataset_size = len(stead_dataset)
    sample_indices = np.random.choice(dataset_size, min(n_samples, dataset_size), replace=False)
    
    samples = []
    for idx in sample_indices:
        sample = load_stead_sample(stead_dataset, idx)
        if sample is not None:
            samples.append(sample)
    
    logger.info(f"Loaded {len(samples)} random samples")
    return samples


def load_balanced_samples(
    stead_dataset,
    n_events: int = 5,
    n_noise: int = 5,
    seed: Optional[int] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load balanced samples (events and noise).
    
    Args:
        stead_dataset: STEADDataset instance
        n_events: Number of earthquake samples
        n_noise: Number of noise samples
        seed: Random seed
    
    Returns:
        Tuple of (event_samples, noise_samples)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Separate by label
    event_indices = []
    noise_indices = []
    
    for idx in range(len(stead_dataset)):
        if hasattr(stead_dataset, 'labels'):
            label = stead_dataset.labels[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
        else:
            # Try to infer from metadata
            _, metadata = stead_dataset.get_sample(idx)
            label = 1 if 'trace_p_arrival_sample' in metadata else 0
        
        if label == 1:
            event_indices.append(idx)
        else:
            noise_indices.append(idx)
    
    # Sample
    event_sample_indices = np.random.choice(
        event_indices,
        min(n_events, len(event_indices)),
        replace=False
    )
    noise_sample_indices = np.random.choice(
        noise_indices,
        min(n_noise, len(noise_indices)),
        replace=False
    )
    
    # Load samples
    event_samples = []
    for idx in event_sample_indices:
        sample = load_stead_sample(stead_dataset, idx)
        if sample is not None:
            event_samples.append(sample)
    
    noise_samples = []
    for idx in noise_sample_indices:
        sample = load_stead_sample(stead_dataset, idx)
        if sample is not None:
            noise_samples.append(sample)
    
    logger.info(f"Loaded {len(event_samples)} event and {len(noise_samples)} noise samples")
    return event_samples, noise_samples


def get_sample_info(sample: Dict) -> str:
    """
    Get formatted info string for a sample.
    
    Args:
        sample: Sample dictionary
    
    Returns:
        Formatted info string
    """
    label = "Event" if sample['label'] == 1 else "Noise"
    info = f"{label}"
    
    if sample.get('metadata'):
        meta = sample['metadata']
        if 'trace_magnitude' in meta:
            info += f" | Mag: {meta['trace_magnitude']:.1f}"
        if 'source_depth' in meta:
            info += f" | Depth: {meta['source_depth']:.0f}km"
        if 'event_id' in meta:
            info += f" | ID: {meta['event_id']}"
    
    return info
