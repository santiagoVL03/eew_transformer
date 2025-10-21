"""
Transformer: Transformer-Based Real-Time Earthquake Detection

Implementation of the Transformer model from Wu et al., 2025.
"""

from .model import Transformer
from .data_loader import STEADLoader
from .preprocessing import preprocess_waveform, extract_phase_window
from .augmentation import WaveformAugmenter
from .trainer import Trainer
from .evaluator import Evaluator
from .utils import set_seed, get_device, count_parameters

__version__ = "1.0.0"
__author__ = "Transformer Implementation"

__all__ = [
    'Transformer',
    'STEADLoader',
    'preprocess_waveform',
    'extract_phase_window',
    'WaveformAugmenter',
    'Trainer',
    'Evaluator',
    'set_seed',
    'get_device',
    'count_parameters',
]
