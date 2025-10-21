# Transformer Architecture and Implementation Details

## Overview

This document explains the architectural of the Tesis

## Architecture Components

### 1. Input Processing

**Specifications:**
- 3-channel seismic waveforms (X, Y, Z components or E, N, Z)
- Default window: 2 seconds @ 100 Hz = 200 samples
- Alternative windows: 4s (400 samples) and 8s (800 samples)

**Implementation:**
```python
# Input shape: (batch_size, 3, seq_len)
# where seq_len = window_size * sampling_rate
# Default: (batch, 3, 200)
```

### 2. Embedding Layer

**Specifications:**
- Linear projection from input channels to model dimension d_model=64
- Maps each timestep from 3 channels to 64-dimensional embedding

**Implementation:**
```python
self.embedding = nn.Linear(input_channels, d_model)
# Input: (batch, seq_len, 3)
# Output: (batch, seq_len, 64)
```

**Why this design:**
- Allows the model to learn rich representations of the 3-component seismic data
- 64 dimensions provide sufficient capacity while keeping the model lightweight
- Each timestep is independently embedded, then temporal relationships are learned by the transformer

### 3. Positional Encoding

**Specifications:**
- Sinusoidal positional encoding at Vaswani et al. (2017)
- Added to embeddings to provide temporal information

**Implementation:**
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Why sinusoidal:**
- Allows the model to easily learn to attend by relative positions
- Works well for sequences of varying lengths
- No additional parameters needed

### 4. Transformer Encoder

**Specifications:**
- Multiple encoder layers (default: 4 layers)
- Each layer contains:
  - Multi-head self-attention (h=8 heads)
  - Position-wise feed-forward network
  - Layer normalization (pre-norm architecture)
  - Residual connections
  - Dropout for regularization

**Implementation:**

```python
# Pre-norm architecture (more stable training):
# x -> LayerNorm -> MultiHeadAttention -> Dropout -> + (residual)
#   -> LayerNorm -> FFN -> Dropout -> + (residual)

class TransformerEncoderLayer(nn.Module):
    def forward(self, x):
        # Pre-norm + MHA + residual
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output)
        
        # Pre-norm + FFN + residual
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output
        
        return x
```

**Multi-Head Attention Details:**
- Number of heads: h = 8
- Dimension per head: d_k = d_model / h = 64 / 8 = 8
- Scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
- Temperature scaling by sqrt(d_k) prevents gradient saturation

**Feed-Forward Network:**
- Input dimension: d_model = 64
- Hidden dimension: d_ff = 256 (4 × d_model, standard transformer ratio)
- Activation: ReLU
- Structure: Linear(64 → 256) → ReLU → Dropout → Linear(256 → 64)

**Why pre-norm instead of post-norm:**
- More stable training, especially for deep networks
- Allows training without warm-up in many cases
- Better gradient flow

### 5. Pooling

**Implementation:**
- Global average pooling across the sequence dimension
- Converts (batch, seq_len, d_model) → (batch, d_model)

**Alternatives considered:**
- Max pooling: Takes maximum activation
- First token (CLS token): Similar to BERT
- Average pooling (chosen): Smoother, less sensitive to outliers

### 6. Classifier (Decoder)

**Specifications:**
- Two fully-connected layers with 200 neurons each
- Final layer outputs logit for binary classification
- Dropout between layers for regularization

**Implementation:**
```python
self.classifier = nn.Sequential(
    nn.Linear(d_model, 200),        # 64 → 200
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(200, 200),             # 200 → 200
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(200, 1)                # 200 → 1
)
```

**Output:**
- Single logit value (before sigmoid)
- BCEWithLogitsLoss used for numerical stability
- Threshold λ = 0.5 applied after sigmoid for binary decision

### 7. MC Dropout for Uncertainty

**Specifications:**
- Dropout active during inference
- Multiple forward passes (N=10) with different dropout masks
- Compute predictive mean and uncertainty (entropy)

**Implementation:**
```python
def predict_with_uncertainty(self, x, n_passes=10):
    self.train()  # Enable dropout
    predictions = []
    
    for _ in range(n_passes):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        predictions.append(probs)
    
    mean_prob = predictions.mean(dim=0)
    
    # Predictive entropy
    entropy = -p*log(p) - (1-p)*log(1-p)
    
    return mean_prob, std_prob, entropy
```

**Why MC Dropout:**
- Provides uncertainty estimates without changing architecture
- High entropy → model is uncertain (useful for real-time detection)
- Low entropy → model is confident
- Helps identify ambiguous cases that may need expert review

**Entropy interpretation:**
- Entropy = 0: Complete certainty (p=0 or p=1)
- Entropy = 1: Maximum uncertainty (p=0.5)
- Can be used to set confidence thresholds for automated detection

## Model Size

**Target:** ~235,000 parameters (as reported in paper)

**Breakdown:**
```
Component                   Parameters
─────────────────────────────────────────
Embedding (3 → 64)          192
Positional Encoding         0 (fixed)
Encoder Layers (×4)         ~180,000
  - MHA (×4)                ~132,000
  - FFN (×4)                ~48,000
  - LayerNorm (×8)          ~1,000
Classifier                  ~53,000
  - FC1 (64 → 200)          13,000
  - FC2 (200 → 200)         40,000
  - FC3 (200 → 1)           200
─────────────────────────────────────────
Total                       ~234,881
```

**Why lightweight:**
- Suitable for real-time deployment
- Can run on edge devices
- Fast inference for early warning systems
- Balances capacity and efficiency

## Training Details

### Loss Function

**Binary Cross-Entropy with Logits:**
```python
loss = -[w_pos * y * log(σ(logit)) + w_neg * (1-y) * log(1-σ(logit))]
```

**Class Weighting:**
- Computed from training set imbalance
- `pos_weight = n_noise / n_earthquake`
- Helps model learn minority class (earthquakes) better
- Prevents bias toward majority class (noise)

### Optimizer

**Adam Optimizer:**
- Learning rate: 1e-4 (default)
- Weight decay: 0 (default, can be adjusted)
- β1=0.9, β2=0.999 (Adam defaults)

**Why Adam:**
- Adaptive learning rates per parameter
- Works well with sparse gradients
- Good default choice for transformers

### Learning Rate Scheduling

**ReduceLROnPlateau (default):**
- Monitors validation loss
- Reduces LR by factor of 0.5 if no improvement for 5 epochs
- Allows model to converge better

**Alternative: CosineAnnealingLR:**
- Smooth annealing from initial LR to minimum
- Good for fixed-epoch training

### Mixed Precision Training

**Automatic Mixed Precision (AMP):**
```python
from torch.cuda.amp import autocast, GradScaler

with autocast():
    logits = model(waveforms)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits:**
- ~2× faster training on modern GPUs (Volta, Turing, Ampere)
- Reduced memory usage
- Same accuracy as FP32

### Early Stopping

- Monitors validation loss
- Patience: 10 epochs (default)
- Prevents overfitting
- Saves best model automatically

## Data Preprocessing

### Bandpass Filtering

**Frequency Range:** 1-30 Hz
- Removes low-frequency noise (< 1 Hz)
- Removes high-frequency noise (> 30 Hz)
- Preserves earthquake signal frequencies

**Implementation:**
```python
from obspy.signal.filter import bandpass

filtered = bandpass(
    waveform,
    freqmin=1.0,
    freqmax=30.0,
    df=sampling_rate,
    corners=4,
    zerophase=True
)
```

**Why zero-phase:**
- No phase distortion
- Preserves signal timing
- Important for P/S wave detection

### Normalization

**Methods:**
1. **Min-Max** (default): Scales to [0, 1] or [-1, 1]
2. **Z-Score**: Zero mean, unit variance
3. **Std**: Divide by standard deviation only

**Applied per-channel:**
- Each component (X, Y, Z) normalized independently
- Preserves relative amplitude relationships

### Window Extraction

**Phase-Centered Windows:**
- Default: 1 second before pick, 1 second after pick
- Total: 2 seconds @ 100 Hz = 200 samples
- Centered on P or S wave arrival

**Padding:**
- If window extends beyond trace, zero-pad
- Ensures consistent input size

## Data Augmentation

### Techniques

1. **Gaussian Noise Addition:**
   - Random SNR between 5-20 dB
   - Simulates varying noise conditions

2. **Amplitude Scaling:**
   - Random factor between 0.8-1.2×
   - Simulates varying magnitudes

3. **Time Shifting:**
   - Random shift ±0.1 seconds
   - Simulates pick uncertainty

**Application:**
- Only applied to training set
- Applied on-the-fly during training
- Probability p=0.5 per augmentation

**Why augmentation:**
- Increases effective training set size
- Improves generalization
- Makes model robust to variations

## Evaluation Metrics

### Classification Metrics

1. **Accuracy:** (TP + TN) / Total
2. **Precision:** TP / (TP + FP) - How many detected earthquakes are real
3. **Recall:** TP / (TP + FN) - How many real earthquakes are detected
4. **F1 Score:** 2 × (Precision × Recall) / (Precision + Recall)
5. **ROC-AUC:** Area under ROC curve

### Uncertainty Metrics

1. **Mean Entropy:** Average uncertainty across predictions
2. **Entropy Distribution:** Correct vs incorrect predictions
3. **Confidence Calibration:** Uncertainty vs prediction confidence

## Offline vs Online Mode

### Offline Mode

**Traditional train/test split:**
- 70% training
- 15% validation
- 15% test
- Model trained to convergence
- Batch processing

**Use case:**
- Research and development
- Model comparison
- Hyperparameter tuning

### Online Mode

**Streaming simulation:**
- Test on sample, then update model
- Simulates real-time operation
- Sequential processing

**Use case:**
- Real-time deployment simulation
- Adaptive learning
- Continual learning scenarios

## Comparison to Paper

### Matches Paper:
✓ 3-channel input (X, Y, Z)  
✓ Embedding dimension: 64  
✓ Multi-head attention: 8 heads  
✓ Positional encoding: Sinusoidal  
✓ Classifier: 2 FC layers with 200 neurons  
✓ MC Dropout for uncertainty  
✓ ~235k parameters  
✓ Bandpass filter: 1-30 Hz  
✓ Binary classification  
✓ Class-weighted loss  

### Implementation Choices:
- Pre-norm vs post-norm (pre-norm chosen for stability)
- Global average pooling (paper doesn't specify pooling method)
- Number of encoder layers: 4 (adjustable, paper doesn't specify exact number)
- FFN hidden dim: 256 (4× d_model, standard ratio)

## References

1. Vaswani et al. (2017). "Attention is All You Need"
2. Mousavi et al. (2019). "STanford EArthquake Dataset (STEAD)"

## Code Organization

```
eew/
├── model.py           # Transformer architecture
├── data_loader.py     # STEAD loading and filtering
├── preprocessing.py   # Signal processing
├── augmentation.py    # Data augmentation
├── trainer.py         # Training loop
├── evaluator.py       # Evaluation metrics
└── utils.py           # Utilities
```

Each module is self-contained and well-documented for easy modification and extension.
