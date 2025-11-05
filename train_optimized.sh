#!/bin/bash
# Optimized training script for STEAD dataset with 48GB RAM
# This script includes all RAM optimizations to prevent crashes

echo "=========================================="
echo "STEAD Transformer Training - RAM Optimized"
echo "=========================================="
echo ""
echo "System Info:"
free -h
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU detected"
echo ""
echo "=========================================="
echo ""

# Activate conda environment
conda activate eew_transformer

# Set environment variables for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

BATCH_SIZE=64
TRAIN_SUBSET=100000
EPOCHS=20
REGION="latam"

echo "Training Configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Training subset: $TRAIN_SUBSET samples"
echo "  Epochs: $EPOCHS"
echo "  Region: $REGION"
echo "  Mode: Lazy loading (memory-efficient)"
echo ""
echo "This configuration uses ~10-20GB RAM instead of 89GB+"
echo ""
read -p "Press Enter to start training (or Ctrl+C to cancel)..."

# Run training with optimized parameters
python run_experiment.py \
    --region $REGION \
    --window 2.0 \
    --batch $BATCH_SIZE \
    --epochs $EPOCHS \
    --lazy_load \
    --train_subset $TRAIN_SUBSET \
    --num_workers 0 \
    --use_gpu \
    --save_dir ./results_final_model \
    2>&1 | tee train.log


echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to ./results"
echo "Full log saved to training_output.log"
echo "=========================================="
