"""
Visualization module for EEW_Transformer.

Provides functions for:
- Full trace visualization with P-wave markers
- Real-time detection simulation
- Individual window prediction visualization
- Grid/mosaic of multiple samples
"""

from .plotter import (
    plot_full_trace_with_P,
    plot_window_prediction,
    plot_grid_of_samples,
    plot_realtime_detection,
    plot_detection_comparison,
    create_professional_figure
)

from .realtime_simulator import (
    simulate_realtime_detection,
    RealTimeSimulator,
    get_detected_windows,
    get_detection_statistics
)

from .data_utils import (
    load_stead_sample,
    extract_window,
    load_random_samples,
    load_balanced_samples,
    get_sample_info
)

__all__ = [
    'plot_full_trace_with_P',
    'plot_window_prediction',
    'plot_grid_of_samples',
    'plot_realtime_detection',
    'plot_detection_comparison',
    'create_professional_figure',
    'simulate_realtime_detection',
    'RealTimeSimulator',
    'get_detected_windows',
    'get_detection_statistics',
    'load_stead_sample',
    'extract_window',
    'load_random_samples',
    'load_balanced_samples',
    'get_sample_info',
]
