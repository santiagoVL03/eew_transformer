"""
Professional plotting functions for seismic waveforms and EEW_Transformer predictions.

Features:
- Full trace visualization with P-wave arrivals
- Individual window predictions with uncertainty
- Grid/mosaic comparisons
- Real-time detection timeline
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Color scheme for professional seismic visualizations
COLORS = {
    'p_wave_real': '#d62728',      # Red: Real P-wave arrival
    'p_wave_predicted': '#1f77b4',  # Blue: Predicted P-wave
    'event': '#2ca02c',             # Green: Seismic event
    'noise': '#ff7f0e',             # Orange: Noise
    'waveform_z': '#000000',        # Black: Z component
    'waveform_e': '#666666',        # Gray: E component
    'waveform_n': '#cccccc',        # Light gray: N component
    'background': '#ffffff',        # White
    'grid': '#e0e0e0',              # Light grid
}

COMPONENT_LABELS = {
    0: 'Z (Vertical)',
    1: 'N (North-South)',
    2: 'E (East-West)',
}


def create_professional_figure(figsize=(12, 8), dpi=100):
    """
    Create a professional matplotlib figure with consistent styling.
    
    Args:
        figsize: Figure size in inches (width, height)
        dpi: Resolution in dots per inch
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Configure styling
    ax.set_facecolor(COLORS['background'])
    ax.grid(True, alpha=0.3, linestyle='--', color=COLORS['grid'])
    
    return fig, ax


def plot_full_trace_with_P(
    waveform: np.ndarray,
    sampling_rate: float = 100.0,
    p_arrival_sample: Optional[int] = None,
    p_arrival_time: Optional[float] = None,
    event_metadata: Optional[Dict] = None,
    title: str = "Seismic Waveform - Full Trace",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8),
    show_components: bool = True
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot full seismic waveform with P-wave arrival marker.
    
    Args:
        waveform: 3-channel waveform (3, n_samples) or (n_samples, 3)
        sampling_rate: Sampling rate in Hz
        p_arrival_sample: Sample index of P-wave arrival
        p_arrival_time: Time in seconds of P-wave arrival
        event_metadata: Dictionary with event info (magnitude, event_id, station, etc.)
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
        show_components: If True, plot each component separately
    
    Returns:
        fig, axes: Matplotlib figure and axes
    """
    # Ensure correct shape: (3, n_samples)
    if waveform.shape[0] != 3:
        waveform = waveform.T
    
    n_samples = waveform.shape[1]
    time_axis = np.arange(n_samples) / sampling_rate
    
    # Create figure with subplots for each component
    fig = plt.figure(figsize=figsize, dpi=100)
    gs = GridSpec(3, 1, figure=fig, hspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    
    # Normalize amplitude for visualization
    amplitudes = [waveform[i] for i in range(3)]
    
    # Plot each component
    for idx, (ax, amplitude) in enumerate(zip(axes, amplitudes)):
        ax.plot(time_axis, amplitude, color=COLORS['waveform_z'], linewidth=0.8, label='Waveform')
        
        # Add P-wave arrival line
        if p_arrival_sample is not None:
            p_time = p_arrival_sample / sampling_rate
            ax.axvline(p_time, color=COLORS['p_wave_real'], linestyle='--', 
                       linewidth=2, label='P-wave arrival (Real)', zorder=3)
        elif p_arrival_time is not None:
            ax.axvline(p_arrival_time, color=COLORS['p_wave_real'], linestyle='--',
                       linewidth=2, label='P-wave arrival (Real)', zorder=3)
        
        # Styling
        ax.set_ylabel(COMPONENT_LABELS.get(idx, f'Channel {idx}'), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':', color=COLORS['grid'])
        ax.set_facecolor(COLORS['background'])
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    # Set common xlabel only on bottom axis
    axes[-1].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    
    # Add title with metadata
    title_text = title
    if event_metadata:
        meta_lines = [
            f"Magnitude: {event_metadata.get('magnitude', 'N/A')}",
            f"Event ID: {event_metadata.get('event_id', 'N/A')}",
            f"Station: {event_metadata.get('station', 'N/A')}"
        ]
        title_text += '\n' + ' | '.join(meta_lines)
    
    fig.suptitle(title_text, fontsize=13, fontweight='bold', y=0.995)
    
    # Save figure if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"Figure saved to {save_path}")
    
    return fig, axes


def plot_window_prediction(
    waveform: np.ndarray,
    prediction: float,
    true_label: int,
    sampling_rate: float = 100.0,
    p_arrival_sample: Optional[int] = None,
    uncertainty: Optional[float] = None,
    window_idx: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot a single 2-second window prediction.
    
    Args:
        waveform: 3-channel waveform (3, 200) or (200, 3)
        prediction: Predicted probability (0-1)
        true_label: True label (0=noise, 1=earthquake)
        sampling_rate: Sampling rate in Hz
        p_arrival_sample: Sample index within window where P-wave starts
        uncertainty: Uncertainty/std from MC dropout
        window_idx: Index of this window in time series
        title: Custom title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    """
    # Ensure correct shape
    if waveform.shape[0] != 3:
        waveform = waveform.T
    
    n_samples = waveform.shape[1]
    time_axis = np.arange(n_samples) / sampling_rate
    
    # Determine prediction class
    pred_label = "EVENT (P-WAVE)" if prediction > 0.5 else "NOISE"
    pred_color = COLORS['event'] if prediction > 0.5 else COLORS['noise']
    true_label_str = "EVENT (P-WAVE)" if true_label == 1 else "NOISE"
    true_color = COLORS['event'] if true_label == 1 else COLORS['noise']
    
    # Correctness indicator
    is_correct = (prediction > 0.5) == (true_label == 1)
    correctness = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    correctness_color = COLORS['event'] if is_correct else COLORS['p_wave_predicted']
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=100)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.2, 1, 1], hspace=0.35)
    
    # Main waveform plot (top, larger)
    ax_main = fig.add_subplot(gs[0])
    
    # Plot all components
    for i, (waveform_ch, label) in enumerate(zip(waveform, COMPONENT_LABELS.values())):
        ax_main.plot(time_axis, waveform_ch, label=label, linewidth=1, alpha=0.8)
    
    # Mark P-wave arrival if present
    if p_arrival_sample is not None:
        p_time = p_arrival_sample / sampling_rate
        ax_main.axvline(p_time, color=COLORS['p_wave_real'], linestyle='--',
                       linewidth=2, label='P-wave in window', zorder=3)
    
    ax_main.set_ylabel('Amplitude', fontsize=10, fontweight='bold')
    ax_main.legend(loc='upper right', fontsize=9, ncol=2)
    ax_main.grid(True, alpha=0.3, linestyle=':', color=COLORS['grid'])
    ax_main.set_facecolor(COLORS['background'])
    
    # Prediction bar plot (bottom left)
    ax_pred = fig.add_subplot(gs[1])
    
    # Probability bar
    ax_pred.barh([0], [prediction], height=0.5, color=pred_color, alpha=0.7, 
                 label='Predicted Probability')
    ax_pred.set_xlim([0, 1])
    ax_pred.set_ylim([-0.5, 0.5])
    ax_pred.set_xlabel('Probability', fontsize=10, fontweight='bold')
    ax_pred.set_yticks([])
    
    # Add probability text
    ax_pred.text(prediction + 0.02, 0, f'{prediction:.3f}', 
                va='center', fontsize=11, fontweight='bold')
    
    if uncertainty is not None:
        # Add uncertainty interval
        ax_pred.errorbar([prediction], [0], xerr=[[uncertainty], [uncertainty]], 
                        fmt='o', color=pred_color, elinewidth=2, capsize=5, markersize=6)
    
    ax_pred.grid(True, alpha=0.2, axis='x', linestyle=':')
    ax_pred.set_facecolor(COLORS['background'])
    
    # Prediction summary (bottom right)
    ax_summary = fig.add_subplot(gs[2])
    ax_summary.axis('off')
    
    summary_text = f"""
    MODEL PREDICTION:  {pred_label}
    Confidence: {prediction:.1%}
    
    TRUE LABEL:  {true_label_str}
    
    {correctness}
    """
    
    ax_summary.text(0.05, 0.5, summary_text, transform=ax_summary.transAxes,
                   fontsize=11, verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor=correctness_color, alpha=0.2))
    
    # Title
    if title is None:
        if window_idx is not None:
            title = f"Window {window_idx}: 2-second Window Prediction"
        else:
            title = "2-second Window Prediction"
    
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
    
    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_realtime_detection(
    full_waveform: np.ndarray,
    window_predictions: List[Dict],
    sampling_rate: float = 100.0,
    p_arrival_sample: Optional[int] = None,
    detection_threshold: float = 0.5,
    title: str = "Real-Time Detection Simulation",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot full waveform with real-time detection overlay.
    
    Args:
        full_waveform: Full waveform (3, n_samples)
        window_predictions: List of dicts with keys:
            - 'start_sample': Start sample of window
            - 'end_sample': End sample of window
            - 'prediction': Predicted probability
            - 'label': True label
        sampling_rate: Sampling rate in Hz
        p_arrival_sample: True P-wave arrival sample
        detection_threshold: Threshold for detection (default 0.5)
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    """
    # Ensure correct shape
    if full_waveform.shape[0] != 3:
        full_waveform = full_waveform.T
    
    n_samples = full_waveform.shape[1]
    time_axis = np.arange(n_samples) / sampling_rate
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=100)
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 1, 1.5], hspace=0.35)
    
    # Plot waveform channels
    for ch_idx in range(3):
        ax = fig.add_subplot(gs[ch_idx])
        ax.plot(time_axis, full_waveform[ch_idx], color=COLORS['waveform_z'], 
               linewidth=0.8, label=COMPONENT_LABELS[ch_idx])
        ax.set_ylabel(COMPONENT_LABELS[ch_idx], fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':', color=COLORS['grid'])
        ax.set_facecolor(COLORS['background'])
        ax.legend(loc='upper right', fontsize=9)
    
    # Detection timeline (bottom)
    ax_detection = fig.add_subplot(gs[3])
    
    # Plot detection windows
    first_detection = None
    for win_pred in window_predictions:
        start_sample = win_pred['start_sample']
        end_sample = win_pred['end_sample']
        prediction = win_pred['prediction']
        
        start_time = start_sample / sampling_rate
        end_time = end_sample / sampling_rate
        
        # Color based on prediction
        if prediction > detection_threshold:
            color = COLORS['event']
            alpha = 0.7
            if first_detection is None:
                first_detection = start_time
        else:
            color = COLORS['noise']
            alpha = 0.2
        
        # Draw detection rectangle
        ax_detection.barh(0, end_time - start_time, left=start_time, height=0.8,
                         color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
    
    # Mark true P-wave arrival
    if p_arrival_sample is not None:
        p_time = p_arrival_sample / sampling_rate
        ax_detection.axvline(p_time, color=COLORS['p_wave_real'], linestyle='--',
                            linewidth=2.5, label='True P-wave Arrival', zorder=5)
    
    # Mark first detection
    if first_detection is not None:
        ax_detection.axvline(first_detection, color=COLORS['p_wave_predicted'],
                            linestyle=':', linewidth=2, label='First Detection', zorder=4)
    
    # Styling
    ax_detection.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax_detection.set_ylabel('Detection', fontsize=10, fontweight='bold')
    ax_detection.set_ylim([-0.5, 0.5])
    ax_detection.set_yticks([])
    ax_detection.grid(True, alpha=0.2, axis='x', linestyle=':')
    ax_detection.set_facecolor(COLORS['background'])
    ax_detection.legend(loc='upper right', fontsize=10)
    
    # Title and labels
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    
    # Common X label
    fig.text(0.5, 0.02, 'Time (seconds)', ha='center', fontsize=11, fontweight='bold')
    
    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_grid_of_samples(
    samples: List[Dict],
    grid_shape: Tuple[int, int] = (3, 3),
    sampling_rate: float = 100.0,
    title: str = "Model Predictions on STEAD Samples",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 12)
) -> plt.Figure:
    """
    Create a grid mosaic of window predictions for comparison.
    
    Args:
        samples: List of dictionaries, each with:
            - 'waveform': (3, 200) or (200, 3) array
            - 'prediction': Predicted probability
            - 'label': True label (0 or 1)
            - 'p_arrival_sample': Optional P-wave arrival sample
            - 'event_id': Optional event identifier
        grid_shape: (rows, cols) for grid layout
        sampling_rate: Sampling rate in Hz
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    """
    rows, cols = grid_shape
    n_samples = min(len(samples), rows * cols)
    
    # Adjust spacing for better centering
    fig = plt.figure(figsize=figsize, dpi=100)
    gs = GridSpec(rows, cols, figure=fig, hspace=0.45, wspace=0.35, 
                  left=0.08, right=0.95, top=0.93, bottom=0.08)
    
    for idx, sample in enumerate(samples[:n_samples]):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(gs[row, col])
        
        # Get waveform
        waveform = sample['waveform']
        if waveform.shape[0] != 3:
            waveform = waveform.T
        
        # Get prediction and label
        prediction = sample.get('prediction', 0.5)
        true_label = sample.get('label', -1)
        p_arrival = sample.get('p_arrival_sample', None)
        
        # Prepare time axis
        n_samples_win = waveform.shape[1]
        time_axis = np.arange(n_samples_win) / sampling_rate
        
        # Plot waveform (average across channels for clarity)
        waveform_avg = waveform.mean(axis=0)
        ax.plot(time_axis, waveform_avg, color=COLORS['waveform_z'], linewidth=1.5)
        ax.fill_between(time_axis, waveform_avg, alpha=0.15, color=COLORS['waveform_z'])
        
        # Mark P-wave if present
        if p_arrival is not None:
            p_time = p_arrival / sampling_rate
            ax.axvline(p_time, color=COLORS['p_wave_real'], linestyle='--', linewidth=2, alpha=0.8)
        
        # Add background color based on prediction
        pred_label = 1 if prediction > 0.5 else 0
        pred_correct = (pred_label == true_label) if true_label >= 0 else None
        
        if pred_correct is True:
            bg_color = COLORS['event']
        elif pred_correct is False:
            bg_color = COLORS['p_wave_predicted']
        else:
            bg_color = 'white'
        
        ax.set_facecolor(bg_color)
        ax.patch.set_alpha(0.08)
        
        # Title with prediction and label
        pred_text = "🔴 EVENT" if prediction > 0.5 else "🔵 NOISE"
        label_text = "✓" if pred_correct is True else ("✗" if pred_correct is False else "?")
        
        title_str = f"{label_text} {pred_text} ({prediction:.2f})"
        if true_label == 1:
            title_str += "\n[TRUE EVENT]"
        elif true_label == 0:
            title_str += "\n[TRUE NOISE]"
        
        ax.set_title(title_str, fontsize=10, fontweight='bold', pad=8)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['0', '1', '2'], fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=9, fontweight='bold')
        ax.set_yticks([])
        ax.grid(True, alpha=0.25, linestyle=':', color=COLORS['grid'])
        
        # Set consistent y-limits
        y_max = np.abs(waveform_avg).max() * 1.1
        if y_max > 0:
            ax.set_ylim([-y_max, y_max])
        
        title_str = f"{label_text} {pred_text} ({prediction:.2f})"
        if true_label == 1:
            title_str += " [TRUE EVENT]"
        elif true_label == 0:
            title_str += " [TRUE NOISE]"
        
        ax.set_title(title_str, fontsize=9, fontweight='bold', pad=8)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['0', '1', '2'], fontsize=8)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_yticks([])
        ax.grid(True, alpha=0.2, linestyle=':', color=COLORS['grid'])
    
    # Figure title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_detection_comparison(
    full_waveform: np.ndarray,
    true_p_arrival_sample: int,
    predicted_detections: List[int],
    sampling_rate: float = 100.0,
    title: str = "True vs Predicted P-wave Detection",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Compare true P-wave arrival with model detections.
    
    Args:
        full_waveform: Full waveform (3, n_samples)
        true_p_arrival_sample: True P-wave arrival sample index
        predicted_detections: List of samples where model detected P-wave
        sampling_rate: Sampling rate in Hz
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    """
    # Ensure correct shape
    if full_waveform.shape[0] != 3:
        full_waveform = full_waveform.T
    
    n_samples = full_waveform.shape[1]
    time_axis = np.arange(n_samples) / sampling_rate
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=100)
    
    # Plot each component
    for ch_idx in range(3):
        ax = plt.subplot(3, 1, ch_idx + 1)
        
        # Plot waveform
        ax.plot(time_axis, full_waveform[ch_idx], color=COLORS['waveform_z'],
               linewidth=0.8, label=COMPONENT_LABELS[ch_idx])
        
        # Mark true P-wave
        true_p_time = true_p_arrival_sample / sampling_rate
        ax.axvline(true_p_time, color=COLORS['p_wave_real'], linestyle='--',
                  linewidth=2.5, label='True P-wave Arrival', zorder=4)
        
        # Mark predicted detections
        if predicted_detections:
            for pred_sample in predicted_detections:
                pred_time = pred_sample / sampling_rate
                ax.axvline(pred_time, color=COLORS['p_wave_predicted'], linestyle=':',
                          linewidth=1.5, alpha=0.7)
        
        ax.set_ylabel(COMPONENT_LABELS[ch_idx], fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':', color=COLORS['grid'])
        ax.set_facecolor(COLORS['background'])
        
        if ch_idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    plt.xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    fig.suptitle(title, fontsize=13, fontweight='bold')
    
    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"Figure saved to {save_path}")
    
    return fig
