#!/bin/bash
# Quick test to verify RAM fix works

echo "============================================"
echo "STEAD Training - RAM FIX VERIFICATION TEST"
echo "============================================"
echo ""
echo "This test will:"
echo "  1. Load STEAD dataset with cache='trace' (memory-efficient)"
echo "  2. Train on 1000 samples for 2 epochs"
echo "  3. Use batch_size=16, num_workers=0"
echo "  4. Should complete in 5-10 minutes using <5GB RAM"
echo ""
echo "Current RAM:"
free -h | grep "Mem:"
echo ""
read -p "Press Enter to start the test (or Ctrl+C to cancel)..."

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eew_transformer

# Set memory-friendly environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run quick test
python run_experiment.py \
    --region latam \
    --window 2.0 \
    --batch 16 \
    --epochs 2 \
    --lazy_load \
    --train_subset 1000 \
    --num_workers 0 \
    --use_gpu \
    --save_dir ./results_ram_test \
    2>&1 | tee ram_test_output.log

echo ""
echo "============================================"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ TEST PASSED!"
    echo "RAM fix is working correctly."
    echo ""
    echo "You can now train on larger datasets:"
    echo "  - For 20k samples: --train_subset 20000"
    echo "  - For 50k samples: --train_subset 50000"
else
    echo "❌ TEST FAILED"
    echo "Check ram_test_output.log for errors"
fi
echo "============================================"
