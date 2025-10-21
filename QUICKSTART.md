# Quick Start Guide

## Installation

### 1. Create Conda Environment

```bash
# Create environment
conda create -n eew_transformer python=3.9 -y
conda activate eew_transformer

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Run tests to verify everything is working
python tests/test_model.py
```

Expected output:
```
Transformer Model Tests
================================================================
Test 1: GPU Availability
✓ GPU available: NVIDIA GeForce RTX 3090
  Device: cuda

Test 2: Model Creation
✓ Model created successfully
  Parameters: 234,881
  ✓ Parameter count is within expected range

... (more tests)

All Tests Passed! ✓
```

## Quick Examples

### Example 1: Basic Training (Chile, 2-second windows)

```bash
python run_experiment.py \
    --region chile \
    --window 2 \
    --batch 64 \
    --epochs 50 \
    --use_gpu \
    --save_dir ./results/chile_2s
```

This will:
1. Download STEAD dataset
2. Filter to Chile events
3. Train Transformer model for 50 epochs
4. Save results to `./results/chile_2s/`

### Example 2: Latin America with Data Augmentation

```bash
python run_experiment.py \
    --region latam \
    --window 2 \
    --batch 64 \
    --epochs 50 \
    --augment \
    --use_gpu \
    --save_dir ./results/latam_2s_aug
```

### Example 3: 4-Second Windows with MC Dropout

```bash
python run_experiment.py \
    --region chile \
    --window 4 \
    --batch 32 \
    --epochs 50 \
    --mc_dropout \
    --n_mc_passes 10 \
    --use_gpu \
    --save_dir ./results/chile_4s_mc
```

### Example 4: Quick Test with Small Dataset

```bash
python run_experiment.py \
    --region chile \
    --window 2 \
    --batch 32 \
    --epochs 10 \
    --max_samples 1000 \
    --use_gpu \
    --save_dir ./results/test
```

### Example 5: Online (Streaming) Mode

```bash
python run_experiment.py \
    --region chile \
    --window 2 \
    --mode online \
    --batch 1 \
    --max_samples 5000 \
    --use_gpu \
    --save_dir ./results/online
```

## Understanding the Results

After running an experiment, you'll find the following in your save directory:

```
results/
├── experiment.log              # Full training log
├── training_history.json       # Training metrics per epoch
├── test_metrics.json          # Final test set metrics
├── checkpoints/
│   └── best_model.pth         # Best model checkpoint
└── plots/
    ├── roc_curve.png          # ROC curve
    ├── confusion_matrix.png   # Confusion matrix
    ├── uncertainty_analysis.png # MC Dropout uncertainty
    └── training_curves.png    # Loss and accuracy curves
```

### Reading Metrics

```python
import json

# Load test metrics
with open('results/chile_2s/test_metrics.json', 'r') as f:
    metrics = json.load(f)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

### Loading Trained Model

```python
import torch
from Transformer.model import create_Transformer_model

# Create model
model = create_Transformer_model(seq_len=200)

# Load checkpoint
checkpoint = torch.load('results/chile_2s/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Make predictions
model.eval()
with torch.no_grad():
    waveform = torch.randn(1, 3, 200)  # Your data
    logit = model(waveform)
    prob = torch.sigmoid(logit)
    
print(f"Earthquake probability: {prob.item():.3f}")
```

## Common Issues and Solutions

### Issue 1: Out of Memory

**Solution:** Reduce batch size

```bash
python run_experiment.py --batch 16  # Instead of 64
```

### Issue 2: GPU Not Detected

**Check CUDA installation:**
```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

**Solution:** Install correct PyTorch version for your CUDA version

### Issue 3: seisbench Download Fails

**Solution:** Use local STEAD files

```bash
python run_experiment.py --data_path /path/to/stead
```

### Issue 4: Training Too Slow

**Solutions:**
- Use mixed precision (enabled by default with `--use_gpu`)
- Reduce number of workers if CPU-bound: `--num_workers 2`
- Reduce max_samples for testing: `--max_samples 5000`

## Advanced Usage

### Custom Hyperparameters

```bash
python run_experiment.py \
    --region chile \
    --window 2 \
    --batch 64 \
    --epochs 100 \
    --lr 0.0005 \
    --dropout 0.2 \
    --num_layers 6 \
    --d_model 128 \
    --scheduler cosine \
    --save_dir ./results/custom
```

### Using Local STEAD Dataset

```bash
# Download STEAD manually, then:
python run_experiment.py \
    --data_path /path/to/stead/data \
    --region chile \
    --window 2 \
    --epochs 50
```

### Resume Training from Checkpoint

```bash
python run_experiment.py \
    --checkpoint ./results/chile_2s/checkpoints/best_model.pth \
    --epochs 100 \
    --save_dir ./results/chile_2s_resumed
```

## Performance Tips

### For Faster Training:

1. **Use GPU with AMP:** `--use_gpu` (enabled by default)
2. **Increase batch size:** `--batch 128` (if GPU memory allows)
3. **Reduce workers if CPU-bound:** `--num_workers 2`
4. **Use CosineAnnealing scheduler:** `--scheduler cosine`

### For Better Accuracy:

1. **Enable augmentation:** `--augment`
2. **Increase epochs:** `--epochs 100`
3. **Use MC Dropout:** `--mc_dropout`
4. **Larger model:** `--num_layers 6 --d_model 128`