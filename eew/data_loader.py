"""
Data loader for STEAD dataset with Chile/Latin America filtering.

Handles:
- STEAD dataset loading via seisbench
- Geographic filtering (Chile/Latin America)
- Train/validation/test splitting
- PyTorch Dataset and DataLoader creation
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import seisbench
    import seisbench.data as sbd
    SEISBENCH_AVAILABLE = True
except ImportError:
    SEISBENCH_AVAILABLE = False
    logging.warning("seisbench not available. Install with: pip install seisbench")


class STEADDataset(Dataset):
    """
    PyTorch Dataset for STEAD waveforms.
    
    Supports two modes:
    1. Preloaded: All data in memory (fast but memory-intensive)
    2. Lazy: Load data on-the-fly (memory-efficient but slower)
    """
    
    def __init__(self, waveforms=None, labels=None, phase_picks=None, metadata=None, 
                 transform=None, seisbench_dataset=None, indices=None, 
                 window_size=2.0, sampling_rate=100, lazy_load=False):
        """
        Args:
            waveforms: Array of waveforms (N, 3, seq_len) - for preloaded mode
            labels: Array of labels (N,) - 1 for earthquake, 0 for noise
            phase_picks: Optional array of phase pick times (N,)
            metadata: Optional metadata dictionary
            transform: Optional transform to apply to waveforms
            seisbench_dataset: Seisbench dataset object - for lazy mode
            indices: List of dataset indices - for lazy mode
            window_size: Window size in seconds (for lazy mode)
            sampling_rate: Sampling rate in Hz (for lazy mode)
            lazy_load: If True, load data on-the-fly instead of preloading
        """
        self.lazy_load = lazy_load
        
        if lazy_load:
            # Lazy loading mode - memory efficient
            if seisbench_dataset is None or indices is None:
                raise ValueError("seisbench_dataset and indices required for lazy loading")
            self.seisbench_dataset = seisbench_dataset
            self.indices = indices
            self.window_size = window_size
            self.sampling_rate = sampling_rate
            self.seq_len = int(window_size * sampling_rate)
            self.labels = labels if labels is not None else [None] * len(indices)
        else:
            # Preloaded mode - all data in memory
            if waveforms is None or labels is None:
                raise ValueError("waveforms and labels required for preloaded mode")
            self.waveforms = waveforms
            self.labels = labels
            self.phase_picks = phase_picks
            self.metadata = metadata
        
        self.transform = transform
    
    def __len__(self):
        if self.lazy_load:
            return len(self.indices)
        else:
            return len(self.waveforms)
    
    def _load_waveform_lazy(self, idx):
        """Load a single waveform on-the-fly from seisbench dataset."""
        dataset_idx = self.indices[idx]
        
        try:
            # Get sample from seisbench - MEMORY OPTIMIZED
            sample = self.seisbench_dataset.get_sample(dataset_idx)
            waveform, metadata = sample
            
            # Convert to numpy if needed and delete torch tensor immediately
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.numpy()
                # Tensor is deleted automatically after conversion
            
            # Ensure 3 components
            if waveform.shape[0] != 3:
                # Pad or truncate to 3 components
                if waveform.shape[0] < 3:
                    waveform = np.pad(waveform, ((0, 3 - waveform.shape[0]), (0, 0)), mode='constant')
                else:
                    waveform = waveform[:3, :]
            
            # Resample if needed - NO RESAMPLING by default to save RAM
            current_sr = 100  # STEAD is typically at 100 Hz
            if current_sr != self.sampling_rate and abs(current_sr - self.sampling_rate) > 1:
                from scipy import signal
                num_samples = int(waveform.shape[1] * self.sampling_rate / current_sr)
                waveform = signal.resample(waveform, num_samples, axis=1)
            
            # Extract window - directly slice to save memory (no intermediate copies)
            if waveform.shape[1] >= self.seq_len:
                center = waveform.shape[1] // 2
                start = max(0, center - self.seq_len // 2)
                end = start + self.seq_len
                waveform = waveform[:, start:end].copy()  # copy to release original
            else:
                # Pad if too short
                pad_width = self.seq_len - waveform.shape[1]
                waveform = np.pad(waveform, ((0, 0), (0, pad_width)), mode='constant')
            
            # CRITICAL: Normalize waveform to prevent NaN issues
            # Apply per-channel normalization to handle varying amplitudes
            for ch in range(waveform.shape[0]):
                std = waveform[ch].std()
                if std > 1e-10:  # Avoid division by zero
                    waveform[ch] = waveform[ch] / std
                # Clip extreme values to prevent numerical instability
                waveform[ch] = np.clip(waveform[ch], -10, 10)
            
            # Get label
            trace_category = metadata.get('trace_category', 'earthquake_local')
            label = 1 if 'earthquake' in str(trace_category).lower() else 0
            
            # Ensure float32 for memory efficiency (not float64)
            return waveform.astype(np.float32), label
            
        except Exception as e:
            # Reduce logging spam
            if idx % 1000 == 0:
                logging.warning(f"Error loading sample {dataset_idx}: {e}")
            # Return zero waveform on error
            return np.zeros((3, self.seq_len), dtype=np.float32), 0
    
    def __getitem__(self, idx):
        if self.lazy_load:
            waveform, label = self._load_waveform_lazy(idx)
        else:
            waveform = self.waveforms[idx]
            label = self.labels[idx]
        
        # Apply transform if provided (data augmentation)
        if self.transform:
            waveform = self.transform(waveform)
        
        # Convert to torch tensors
        waveform = torch.FloatTensor(waveform)
        label = torch.FloatTensor([label])
        
        return waveform, label


class STEADLoader:
    """
    STEAD dataset loader with filtering capabilities.
    """
    
    # Geographic bounding boxes
    REGIONS = {
        'chile': {
            'lat_min': -56.0,
            'lat_max': -17.0,
            'lon_min': -76.0,
            'lon_max': -66.0,
            'networks': ['C', 'C1', 'CX'],  # Chilean networks
        },
        'latam': {
            'lat_min': -60.0,
            'lat_max': 15.0,
            'lon_min': -120.0,
            'lon_max': -30.0,
            'networks': None,  # Use coordinates only
        }
    }
    
    def __init__(self, data_path=None, region='chile', cache_dir=None):
        """
        Args:
            data_path: Optional path to local STEAD files
            region: Region to filter ('chile' or 'latam')
            cache_dir: Directory to cache downloaded data
        """
        self.data_path = data_path
        self.region = region.lower()
        self.cache_dir = cache_dir or './data/stead'
        
        if self.region not in self.REGIONS:
            raise ValueError(f"Region must be 'chile' or 'latam', got '{region}'")
        
        self.dataset = None
        self.filtered_indices = None
        self.metadata = None
    
    def load_dataset(self):
        """
        Load STEAD dataset using seisbench.
        
        Returns:
            seisbench dataset object
        """
        if not SEISBENCH_AVAILABLE:
            raise RuntimeError("seisbench is required. Install with: pip install seisbench")
        
        logging.info("Loading STEAD dataset...")
        
        # CRITICAL FIX: Use cache='trace' instead of cache='full'
        # 'full' loads 86GB metadata into RAM
        # 'trace' only loads metadata index and keeps waveforms on disk
        
        if self.data_path:
            # Load from local path
            logging.info(f"Loading from local path: {self.data_path}")
            logging.info("Using cache='trace' mode (memory-efficient, keeps waveforms on disk)")
            self.dataset = sbd.STEAD(path=self.data_path, cache='trace')
        else:
            # Download dataset - seisbench will use default cache location
            logging.info(f"Downloading STEAD dataset (will be cached by seisbench)")
            logging.info(f"Note: Data will be cached in seisbench's default location")
            logging.info("Using cache='trace' mode (memory-efficient, keeps waveforms on disk)")
            # Use 'trace' cache to avoid loading 86GB into RAM
            self.dataset = sbd.STEAD(cache='trace')
        
        logging.info(f"STEAD dataset loaded: {len(self.dataset)} samples")
        logging.info("Metadata table loaded (waveforms stay on disk)")
        
        return self.dataset
    
    def filter_by_region(self):
        """
        Filter STEAD dataset by geographic region.
        
        Returns:
            Filtered indices
        """
        if self.dataset is None:
            self.load_dataset()
        
        region_config = self.REGIONS[self.region]
        logging.info(f"Filtering by region: {self.region}")
        logging.info(f"Bounding box: lat=[{region_config['lat_min']}, {region_config['lat_max']}], "
                    f"lon=[{region_config['lon_min']}, {region_config['lon_max']}]")
        
        # Get metadata
        metadata = self.dataset.metadata
        
        # Try different filtering approaches
        filtered_indices = []
        
        # Approach 1: Filter by source coordinates (preferred)
        if 'source_latitude' in metadata.columns and 'source_longitude' in metadata.columns:
            logging.info("Filtering by source coordinates...")
            mask = (
                (metadata['source_latitude'] >= region_config['lat_min']) &
                (metadata['source_latitude'] <= region_config['lat_max']) &
                (metadata['source_longitude'] >= region_config['lon_min']) &
                (metadata['source_longitude'] <= region_config['lon_max'])
            )
            filtered_indices = metadata[mask].index.tolist()
        
        # Approach 2: Filter by station coordinates (fallback)
        elif 'station_latitude' in metadata.columns and 'station_longitude' in metadata.columns:
            logging.info("Filtering by station coordinates (fallback)...")
            mask = (
                (metadata['station_latitude'] >= region_config['lat_min']) &
                (metadata['station_latitude'] <= region_config['lat_max']) &
                (metadata['station_longitude'] >= region_config['lon_min']) &
                (metadata['station_longitude'] <= region_config['lon_max'])
            )
            filtered_indices = metadata[mask].index.tolist()
        
        # Approach 3: Filter by network code
        if region_config['networks'] and 'network_code' in metadata.columns:
            logging.info(f"Additional filtering by network codes: {region_config['networks']}")
            network_mask = metadata['network_code'].isin(region_config['networks'])
            network_indices = metadata[network_mask].index.tolist()
            
            # Combine with coordinate filtering if available
            if filtered_indices:
                filtered_indices = list(set(filtered_indices) | set(network_indices))
            else:
                filtered_indices = network_indices
        
        if not filtered_indices:
            logging.warning("No samples found for the specified region! Check metadata columns.")
            logging.info(f"Available metadata columns: {list(metadata.columns)}")
            # Return all indices as fallback
            filtered_indices = list(range(len(metadata)))
        
        self.filtered_indices = filtered_indices
        logging.info(f"Filtered to {len(filtered_indices)} samples for region '{self.region}'")
        
        return filtered_indices
    
    def load_waveforms(
        self,
        indices=None,
        phase='P',
        window_size=2.0,
        sampling_rate=100,
        max_samples=None
    ):
        """
        Load waveforms for specified indices.
        
        Args:
            indices: List of indices to load (if None, use filtered_indices)
            phase: Phase to extract ('P', 'S', or 'both')
            window_size: Window size in seconds (default 2.0)
            sampling_rate: Target sampling rate in Hz (default 100)
            max_samples: Maximum number of samples to load (for testing)
        
        Returns:
            waveforms: Array of waveforms (N, 3, seq_len)
            labels: Array of labels (N,)
            metadata: Dictionary with sample metadata
        """
        if self.dataset is None:
            self.load_dataset()
        
        if indices is None:
            if self.filtered_indices is None:
                self.filter_by_region()
            indices = self.filtered_indices
        
        if indices is None:
            raise ValueError("No indices available to load waveforms")
        
        if max_samples:
            indices = indices[:max_samples]
        
        logging.info(f"Loading {len(indices)} waveforms (phase={phase}, window={window_size}s)...")
        
        waveforms_list = []
        labels_list = []
        metadata_list = []
        
        seq_len = int(window_size * sampling_rate)
        
        for i, idx in enumerate(indices):
            if i % 1000 == 0:
                logging.info(f"Loading... {i}/{len(indices)}")
            
            try:
                # Get sample from seisbench dataset
                # seisbench returns (waveform, metadata_dict) tuple
                waveform, metadata = self.dataset.get_sample(idx)
                
                # Waveform is already a numpy array
                if isinstance(waveform, torch.Tensor):
                    waveform = waveform.numpy()
                
                # Ensure 3 components
                if waveform.shape[0] != 3:
                    logging.warning(f"Sample {idx} has {waveform.shape[0]} components, skipping")
                    continue
                
                # Resample if needed
                current_sr = 100  # STEAD is typically at 100 Hz
                if current_sr != sampling_rate:
                    from scipy import signal
                    num_samples = int(waveform.shape[1] * sampling_rate / current_sr)
                    waveform = signal.resample(waveform, num_samples, axis=1)
                
                # Extract window around phase pick
                # For now, use center of trace if phase pick not available
                if waveform.shape[1] >= seq_len:
                    center = waveform.shape[1] // 2
                    start = max(0, center - seq_len // 2)
                    end = start + seq_len
                    waveform = waveform[:, start:end]
                else:
                    # Pad if too short
                    pad_width = seq_len - waveform.shape[1]
                    waveform = np.pad(waveform, ((0, 0), (0, pad_width)), mode='constant')
                
                # CRITICAL: Normalize waveform to prevent NaN issues
                # Apply per-channel normalization to handle varying amplitudes
                for ch in range(waveform.shape[0]):
                    std = waveform[ch].std()
                    if std > 1e-10:  # Avoid division by zero
                        waveform[ch] = waveform[ch] / std
                    # Clip extreme values to prevent numerical instability
                    waveform[ch] = np.clip(waveform[ch], -10, 10)
                
                # Label: 1 for earthquake, 0 for noise
                # In STEAD, trace_category indicates earthquake vs noise
                trace_category = metadata.get('trace_category', 'earthquake_local')
                label = 1 if 'earthquake' in str(trace_category).lower() else 0
                
                waveforms_list.append(waveform)
                labels_list.append(label)
                metadata_list.append({
                    'index': idx,
                    'trace_name': metadata.get('trace_name', ''),
                })
            
            except Exception as e:
                logging.warning(f"Error loading sample {idx}: {e}")
                continue
        
        waveforms = np.array(waveforms_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int64)
        
        logging.info(f"Loaded {len(waveforms)} waveforms successfully")
        logging.info(f"Waveforms shape: {waveforms.shape}")
        logging.info(f"Labels: {np.sum(labels)} earthquakes, {len(labels) - np.sum(labels)} noise")
        
        return waveforms, labels, metadata_list
    
    def create_dataloaders(
        self,
        waveforms=None,
        labels=None,
        batch_size=64,
        train_ratio=0.7,
        val_ratio=0.15,
        shuffle=True,
        num_workers=4,
        transform_train=None,
        transform_val=None,
        lazy_load=False,
        window_size=2.0,
        sampling_rate=100
    ):
        """
        Create train/val/test dataloaders.
        
        Args:
            waveforms: Array of waveforms (None if lazy_load=True)
            labels: Array of labels (None if lazy_load=True)
            batch_size: Batch size
            train_ratio: Training set ratio
            val_ratio: Validation set ratio (test = 1 - train - val)
            shuffle: Whether to shuffle data
            num_workers: Number of dataloader workers
            transform_train: Transform for training set
            transform_val: Transform for val/test sets
            lazy_load: If True, load data on-the-fly (memory-efficient)
            window_size: Window size in seconds (for lazy loading)
            sampling_rate: Sampling rate in Hz (for lazy loading)
        
        Returns:
            train_loader, val_loader, test_loader
        """
        if lazy_load:
            # Use filtered indices for lazy loading
            if self.filtered_indices is None:
                self.filter_by_region()
            
            all_indices = np.array(self.filtered_indices)
            n_samples = len(all_indices)
        else:
            # Use preloaded waveforms
            if waveforms is None or labels is None:
                raise ValueError("waveforms and labels required when lazy_load=False")
            n_samples = len(waveforms)
            all_indices = np.arange(n_samples)
        
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        # Split indices
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Create datasets
        if lazy_load:
            # Lazy loading mode - memory efficient
            train_dataset = STEADDataset(
                seisbench_dataset=self.dataset,
                indices=all_indices[train_indices].tolist(),
                window_size=window_size,
                sampling_rate=sampling_rate,
                transform=transform_train,
                lazy_load=True
            )
            
            val_dataset = STEADDataset(
                seisbench_dataset=self.dataset,
                indices=all_indices[val_indices].tolist(),
                window_size=window_size,
                sampling_rate=sampling_rate,
                transform=transform_val,
                lazy_load=True
            )
            
            test_dataset = STEADDataset(
                seisbench_dataset=self.dataset,
                indices=all_indices[test_indices].tolist(),
                window_size=window_size,
                sampling_rate=sampling_rate,
                transform=transform_val,
                lazy_load=True
            )
        else:
            # Preloaded mode - all data in memory
            train_dataset = STEADDataset(
                waveforms[train_indices],
                labels[train_indices],
                transform=transform_train,
                lazy_load=False
            )
            
            val_dataset = STEADDataset(
                waveforms[val_indices],
                labels[val_indices],
                transform=transform_val,
                lazy_load=False
            )
            
            test_dataset = STEADDataset(
                waveforms[test_indices],
                labels[test_indices],
                transform=transform_val,
                lazy_load=False
            )
        
        # Create dataloaders with memory-efficient settings
        # For large datasets like STEAD (89GB), we optimize for RAM usage:
        # - persistent_workers=False to not keep workers in memory between epochs
        # - prefetch_factor=2 (default) to limit memory used for prefetching
        # - pin_memory=True only if not using lazy loading (to speed up GPU transfer)
        
        use_pin_memory = (not lazy_load) and torch.cuda.is_available()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=False,  # Don't keep workers alive to save RAM
            prefetch_factor=2 if num_workers > 0 else None,  # Limit prefetch buffer
            drop_last=True  # Drop incomplete batches to maintain consistent batch size
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=False,  # Don't keep workers alive to save RAM
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=False,  # Don't keep workers alive to save RAM
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=False
        )
        
        logging.info(f"Created dataloaders (lazy_load={lazy_load}):")
        logging.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        logging.info(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
        logging.info(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test STEAD loader
    logging.basicConfig(level=logging.INFO)
    
    loader = STEADLoader(region='chile')
    loader.load_dataset()
    loader.filter_by_region()
    
    # Load small sample
    waveforms, labels, metadata = loader.load_waveforms(max_samples=100)
    
    print(f"\nTest successful!")
    print(f"Loaded {len(waveforms)} waveforms")
    print(f"Waveform shape: {waveforms[0].shape}")
