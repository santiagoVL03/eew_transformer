# Transformer: Transformer-Based Real-Time Earthquake Detection

Implementation of the Transformer model from "A transformer-based real-time earthquake detection framework in heterogeneous environments" (Wu et al., 2025).

## Overview

This project implements a transformer-based earthquake detection model trained on the STEAD dataset, filtered for Chile and Latin America events. The model uses MC Dropout for uncertainty quantification and supports both offline and online (streaming) evaluation modes.

## Features

- **Transformer Transformer Architecture**: 
  - Input: 3-channel (X, Y, Z) seismic waveforms
  - Embedding dimension: 64
  - Multi-head attention: 8 heads
  - MC Dropout for uncertainty estimation
  - ~235k parameters (lightweight)

- **Data Processing**:
  - STEAD dataset loading via seisbench
  - Geographic filtering (Chile/Latin America)
  - Bandpass filtering (1-30 Hz)
  - Phase-centered windowing (2s/4s/8s)
  - Data augmentation (noise, scaling, time shift)

- **Training Features**:
  - GPU-accelerated with mixed precision (AMP)
  - Class-weighted loss for imbalanced data
  - Early stopping and checkpointing
  - Learning rate scheduling
  - Comprehensive logging

- **Evaluation**:
  - Accuracy, Precision, Recall, F1, ROC-AUC
  - MC Dropout uncertainty quantification
  - Offline and online (streaming) modes

## Installation

### Option 1: Conda Environment (Recommended)

```bash
# Create environment
conda create -n eew_transformer python=3.9 -y
conda activate eew_transformer

# Install PyTorch (GPU version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt
```

### Option 2: pip only

```bash
pip install -r requirements.txt
```

**Note**: Make sure to install the GPU version of PyTorch appropriate for your CUDA version. Visit [pytorch.org](https://pytorch.org) for installation commands.

## Quick Start

### 1. Basic Training (2-second windows, Chile region)

```bash
python run_experiment.py --region chile --window 2 --batch 64 --epochs 50
```

### 2. Latin America region with 4-second windows

```bash
python run_experiment.py --region latam --window 4 --batch 32 --epochs 50
```

### 3. Full experiment with all options

```bash
python run_experiment.py \
    --region chile \
    --window 2 \
    --batch 64 \
    --epochs 100 \
    --lr 0.0001 \
    --phase both \
    --mode offline \
    --augment \
    --use_gpu \
    --save_dir ./results
```

## Command Line Arguments

```
--dataset        Dataset to use (default: stead)
--data_path      Path to local STEAD files (optional, will download if not provided)
--region         Region filter: 'chile' or 'latam' (default: chile)
--window         Window size in seconds: 2, 4, or 8 (default: 2)
--phase          Phase to use: 'P', 'S', or 'both' (default: both)
--mode           Training mode: 'offline' or 'online' (default: offline)
--batch          Batch size (default: 64)
--epochs         Number of epochs (default: 50)
--lr             Learning rate (default: 0.0001)
--dropout        Dropout rate (default: 0.1)
--augment        Enable data augmentation
--use_gpu        Use GPU if available
--seed           Random seed for reproducibility (default: 42)
--save_dir       Directory to save results (default: ./results)
--checkpoint     Path to checkpoint to resume training
```

## Dataset Filtering

The code filters STEAD events to Chile/Latin America using:

1. **Metadata-based filtering** (preferred):
   - Country code (`metadata['country'] == 'CL'`)
   - Network codes (e.g., 'C', 'C1' for Chile)

2. **Geographic bounding box** (fallback):
   - **Chile**: lat ∈ [-56.0, -17.0], lon ∈ [-76.0, -66.0]
   - **Latin America**: lat ∈ [-60.0, 15.0], lon ∈ [-120.0, -30.0]

## Project Structure

```
eew_transformer/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment file
├── run_experiment.py         # Main experiment script
├── eew/
│   ├── __init__.py
│   ├── model.py             # Transformer transformer architecture
│   ├── data_loader.py       # STEAD loading and filtering
│   ├── preprocessing.py     # Signal preprocessing
│   ├── augmentation.py      # Data augmentation
│   ├── trainer.py           # Training loop
│   ├── evaluator.py         # Evaluation metrics
│   └── utils.py             # Utilities (logging, seeds, etc.)
└── tests/
    └── test_model.py        # Unit tests
```

## Architecture Details

### Transformer Model

Following the paper (Wu et al., 2025):

1. **Input**: 3-channel waveform (X, Y, Z) × n timesteps (n=200 for 2s @ 100Hz)
2. **Embedding**: Linear projection (3 → 64) per timestep
3. **Positional Encoding**: Sinusoidal (Vaswani et al.)
4. **Transformer Encoder**:
   - Multi-head attention (h=8 heads, d_k=8)
   - Feed-forward network (d_model=64, d_ff=256)
   - Layer normalization (pre-norm architecture)
   - Dropout for regularization
5. **Classifier (Decoder)**:
   - 2 fully-connected layers (64 → 200 → 200 → 1)
   - MC Dropout enabled at inference
   - Binary classification (earthquake vs noise)

### MC Dropout Uncertainty

The model performs N=10 stochastic forward passes with dropout enabled at inference to compute:
- **Predictive mean**: Average probability across passes
- **Predictive entropy**: Uncertainty measure from probability distribution

Higher entropy indicates higher model uncertainty.

## Training Details

- **Optimizer**: Adam (lr=1e-4)
- **Loss**: Binary cross-entropy with class weights
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Mixed Precision**: Enabled for speed (torch.cuda.amp)
- **Early Stopping**: Patience=10 epochs on validation loss
- **Data Split**: 70% train, 15% validation, 15% test (offline mode)

## Preprocessing Pipeline

1. **Bandpass Filter**: 1-30 Hz (ObsPy)
2. **Resampling**: To 100 Hz if needed
3. **Normalization**: Per-trace min-max or z-score
4. **Windowing**: Phase-centered (1s before, 1s after pick)
5. **Augmentation** (optional):
   - Gaussian noise (SNR 5-20 dB)
   - Amplitude scaling (0.8-1.2×)
   - Time shifting (±0.1s)

## Reproducibility

Set random seed for reproducibility:

```bash
python run_experiment.py --seed 42
```

**Note**: Full determinism on GPU requires additional settings that may reduce performance:
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

These are enabled when a seed is set.

## Expected Output

The script will:

1. Download/load STEAD dataset
2. Filter to selected region
3. Preprocess waveforms
4. Train Transformer model with progress bars
5. Evaluate on test set
6. Save:
   - Model checkpoint (`best_model.pth`)
   - Training history (`training_history.json`)
   - Metrics (`test_metrics.json`)
   - Plots:
     - ROC curve (`roc_curve.png`)
     - Training curves (`training_curves.png`)
     - Uncertainty analysis (`uncertainty_analysis.png`)

### Sample Output

```
=== Transformer Earthquake Detection Experiment ===
Region: Chile
Window: 2.0s (200 samples)
Device: cuda

Loading STEAD dataset...
Filtered to 12,453 Chile events
Train: 8,717 | Val: 1,868 | Test: 1,868

Model Parameters: 234,881

Training...
Epoch 1/50: 100%|████████| Loss: 0.456 | Val Loss: 0.389 | Val Acc: 0.842
...
Best model saved at epoch 23

Evaluating...
Test Accuracy: 0.891
Test Precision: 0.876
Test Recall: 0.903
Test F1: 0.889
Test ROC-AUC: 0.945

MC Dropout Uncertainty: Mean entropy = 0.234
```

## GPU Usage

The code automatically detects and uses GPU if available. To force CPU:

```bash
python run_experiment.py --use_gpu False
```

Check GPU usage:

```bash
nvidia-smi
```

## Testing

Run unit tests to verify installation:

```bash
python tests/test_model.py
```

This will check:
- GPU availability
- Model architecture and parameter count
- Forward pass shapes
- Data loading pipeline

## References

- **STEAD Dataset**: Mousavi et al. (2019)
- **Seisbench**: Woollam et al. (2022)