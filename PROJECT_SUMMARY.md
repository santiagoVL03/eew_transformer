# Transformer Implementation - Project Summary

## Overview

The project implements a transformer-based real-time earthquake detection system trained on the STEAD dataset, filtered for Chile and Latin America events.

## Deliverables Completed

### Core Implementation

- **Transformer Model** (`eew/model.py`)
  - 3-channel input processing (X, Y, Z seismic components)
  - Linear embedding (3 → 64 dimensions)
  - Sinusoidal positional encoding
  - Multi-head attention (8 heads)
  - 4 transformer encoder layers
  - MLP classifier with MC Dropout
  - ~235k parameters

- **Data Loader** (`eew/data_loader.py`)
  - STEAD dataset loading via seisbench
  - Geographic filtering for differents Regions
  - Metadata-based and bounding-box filtering
  - Train/val/test splitting (70/15/15)
  - PyTorch Dataset and DataLoader integration

- **Preprocessing Pipeline** (`eew/preprocessing.py`)
  - Bandpass filtering (1-30 Hz) using ObsPy/SciPy
  - Resampling to 100 Hz
  - Multiple normalization methods (min-max, z-score, std)
  - Phase-centered window extraction
  - Support for 2s/4s/8s windows

- **Data Augmentation** (`eew/augmentation.py`)
  - Gaussian noise addition (SNR 5-20 dB)
  - Amplitude scaling (0.8-1.2×)
  - Time shifting (±0.1s)
  - On-the-fly augmentation during training

- **Training System** (`eew/trainer.py`)
  - GPU-accelerated with mixed precision (AMP)
  - Class-weighted loss for imbalanced data
  - Adam optimizer with learning rate scheduling
  - Early stopping and checkpointing
  - Progress tracking with tqdm
  - Offline and online (streaming) modes

- **Evaluation Suite** (`eew/evaluator.py`)
  - Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
  - MC Dropout uncertainty quantification
  - Visualization (ROC curves, confusion matrix, uncertainty plots)
  - Classification reports

### Utilities and Tools

- **Utility Functions** (`eew/utils.py`)
  - Seed setting for reproducibility
  - Device management (GPU/CPU)
  - Parameter counting
  - Checkpoint saving/loading
  - Early stopping
  - Logging setup

- **Main Experiment Script** (`run_experiment.py`)
  - Command-line interface with argparse
  - Full experiment pipeline
  - Configurable hyperparameters
  - Automatic result saving
  - Comprehensive logging

- **Demo Script** (`demo.py`)
  - Interactive inference demonstration
  - Synthetic waveform generation
  - Visualization of predictions
  - MC Dropout uncertainty display

- **Unit Tests** (`tests/test_model.py`)
  - GPU availability check
  - Model creation and parameter count
  - Forward pass validation
  - MC Dropout testing
  - Preprocessing pipeline
  - Data augmentation
  - Reproducibility verification

### Documentation

- **README.md** - Main documentation
- **ARCHITECTURE.md** - Detailed architecture explanation
- **QUICKSTART.md** - Quick start guide and examples
- **requirements.txt** - Python dependencies
- **environment.yml** - Conda environment specification
- **.gitignore** - Git ignore patterns

## Key Features

### Model Architecture (Following Wu et al., 2025)

1. **Input Processing**
   - 3-channel seismic waveforms (X, Y, Z)
   - Default: 200 samples (2s @ 100Hz)
   - Support for 400 and 800 samples (4s, 8s)

2. **Embedding & Encoding**
   - Linear projection: 3 → 64 dimensions
   - Sinusoidal positional encoding
   - Preserves temporal information

3. **Transformer Encoder**
   - 4 layers (configurable)
   - Multi-head attention (8 heads)
   - Feed-forward networks (256 hidden units)
   - Pre-norm architecture for stability
   - Dropout for regularization

4. **Classifier**
   - 2 FC layers (64 → 200 → 200 → 1)
   - MC Dropout for uncertainty
   - Binary classification (earthquake vs noise)

5. **MC Dropout Uncertainty**
   - N=10 stochastic forward passes
   - Predictive mean and entropy
   - Uncertainty quantification for real-time decisions

### Training Features

- **GPU Acceleration**: CUDA support with mixed precision
- **Class Balancing**: Weighted loss for imbalanced data
- **Optimization**: Adam optimizer with LR scheduling
- **Regularization**: Dropout, early stopping
- **Monitoring**: Real-time progress bars, comprehensive logging
- **Checkpointing**: Best model saving, resumable training

### Data Features

- **STEAD Integration**: Automatic download via seisbench
- **Geographic Filtering**:
  - Chile: lat [-56, -17], lon [-76, -66]
  - Latin America: lat [-60, 15], lon [-120, -30]
- **Preprocessing**: Bandpass filtering, normalization
- **Augmentation**: Noise, scaling, time shifting
- **Efficient Loading**: Multi-worker data loading, pin memory

### Evaluation Features

- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Uncertainty Analysis**: MC Dropout entropy quantification
- **Visualizations**:
  - ROC curves
  - Confusion matrices
  - Training curves
  - Uncertainty distributions
- **Reports**: JSON metrics, classification reports

## Quick Start

### Installation

```bash
# Create conda environment
conda create -n eew_transformer python=3.9 -y
conda activate eew_transformer

# Install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

### Run Tests

```bash
python tests/test_model.py
```

### Train Model

```bash
python run_experiment.py \
    --region chile \
    --window 2 \
    --batch 64 \
    --epochs 50 \
    --augment \
    --mc_dropout \
    --use_gpu \
    --save_dir ./results/chile_2s
```

### Run Demo

```bash
python demo.py \
    --checkpoint results/chile_2s/checkpoints/best_model.pth \
    --mc_dropout \
    --save_plots
```

## Project Structure

```
eew_transformer/
├── README.md                 # Main documentation
├── ARCHITECTURE.md           # Architecture details
├── QUICKSTART.md            # Quick start guide
├── LICENSE                  # MIT License
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment
├── .gitignore              # Git ignore patterns
│
├── run_experiment.py        # Main experiment script
├── demo.py                  # Inference demo
│
├── eew/                    # Main package
│   ├── __init__.py
│   ├── model.py            # Transformer transformer model
│   ├── data_loader.py      # STEAD data loading
│   ├── preprocessing.py    # Signal preprocessing
│   ├── augmentation.py     # Data augmentation
│   ├── trainer.py          # Training loop
│   ├── evaluator.py        # Evaluation metrics
│   └── utils.py            # Utility functions
│
└── tests/
    └── test_model.py        # Unit tests
```

## Implementation Notes

### Design Decisions

1. **Pre-norm vs Post-norm**: Used pre-norm for better training stability
2. **Pooling Method**: Global average pooling for robustness
3. **Loss Function**: BCEWithLogitsLoss with class weights
4. **Scheduler**: ReduceLROnPlateau for adaptive learning rate
5. **AMP**: Mixed precision for 2× speedup on modern GPUs

### Extras

- Exact number of encoder layers not specified in paper (used 4)
- Pooling method not specified (used global average)

### Matching Specifications

Input: 3-channel waveforms  
Embedding: d_model = 64  
Attention: 8 heads  
Positional encoding: Sinusoidal  
Classifier: 2 FC layers with 200 neurons  
MC Dropout: N=10 passes  
Parameters: ~235k  
Bandpass: 1-30 Hz  
Windows: 2s/4s/8s  
Class weighting: Binary cross-entropy  

## Usage Examples

### Example 1: Basic Training

```bash
python run_experiment.py --region chile --window 2 --epochs 50
```

### Example 2: With Augmentation

```bash
python run_experiment.py --region latam --window 4 --augment --epochs 100
```

### Example 3: Online Mode

```bash
python run_experiment.py --mode online --max_samples 5000
```

### Example 4: Custom Hyperparameters

```bash
python run_experiment.py \
    --d_model 128 \
    --num_layers 6 \
    --dropout 0.2 \
    --lr 0.0005 \
    --batch 128
```

## Customization

### Modify Model Architecture

Edit `eew/model.py`:
```python
model = create_Transformer_model(
    seq_len=200,
    d_model=128,      # Change embedding dimension
    nhead=16,         # Change number of heads
    num_encoder_layers=6,  # Change depth
    dropout=0.2       # Change dropout rate
)
```

### Add New Data Augmentation

Edit `eew/augmentation.py`:
```python
def _new_augmentation(self, waveform):
    # Implement new augmentation
    return augmented_waveform
```

### Change Loss Function

Edit `eew/trainer.py`:
```python
self.criterion = nn.CustomLoss(...)
```

## Results Structure

After training, results are saved to:

```
results/
├── experiment.log              # Full log
├── training_history.json       # Training metrics
├── test_metrics.json          # Test metrics
├── checkpoints/
│   └── best_model.pth         # Best model
└── plots/
    ├── roc_curve.png
    ├── confusion_matrix.png
    ├── training_curves.png
    └── uncertainty_analysis.png
```

## Troubleshooting

### GPU Not Available

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU
python run_experiment.py --device cpu
```

### Out of Memory

```bash
# Reduce batch size
python run_experiment.py --batch 16

# Or use gradient accumulation (manual implementation needed)
```

### Slow Training

```bash
# Reduce num_workers if CPU-bound
python run_experiment.py --num_workers 2

# Use smaller dataset for testing
python run_experiment.py --max_samples 1000
```

## References

1. **Mousavi et al. (2019)**. "STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI"

2. **Vaswani et al. (2017)**. "Attention is All You Need"

## ✨ Summary

This is a **complete, production-ready** implementation of a Transformer model:

- Full model architecture
- STEAD data loading and filtering
- Comprehensive preprocessing pipeline
- Data augmentation
- GPU-accelerated training
- MC Dropout uncertainty quantification
- Extensive evaluation metrics
- Visualization tools
- Unit tests
- Complete documentation
- Example scripts
- Command-line interface

**Ready to use for research and deployment!**
