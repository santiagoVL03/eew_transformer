#!/bin/bash

# ENTRENAMIENTO MEJORADO CON TODOS LOS FIXES
# Basado en análisis profundo del por qué no se alcanza 95%

echo "============================================================"
echo "ENTRENAMIENTO MEJORADO - Todas las Optimizaciones"
echo "============================================================"
echo ""
echo "Mejoras aplicadas:"
echo "  ✅ Normalización Z-score correcta (mean - mean, divide por std)"
echo "  ✅ P-wave alignment (alinea ventana con P-wave arrival)"
echo "  ✅ Modelo más grande (d_model 128, 6 layers)"
echo "  ✅ Bigger classifier (256 hidden)"
echo "  ✅ SNR más realista (10-25 dB en lugar de 5-20)"
echo "  ✅ Mejor dropout (0.15 global)"
echo "  ✅ Cosine scheduler con warmup"
echo ""
echo "Impacto estimado:"
echo "  - Normalización: +5-8%"
echo "  - P-wave alignment: +10-15%"
echo "  - Modelo más grande: +3-5%"
echo "  - Total estimado: +30-40%"
echo "  - Baseline actual: 59%"
echo "  - Expected con mejoras: 85-99%"
echo ""
echo "============================================================"
echo ""

python run_experiment.py \
    --lazy_load \
    --region latam \
    --augment \
    --balanced_aug \
    --epochs 60 \
    --batch 64 \
    --save_dir ./results_improved_v2 \
    --threshold 0.5 \
    --loss_type bce \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --d_model 128 \
    --num_layers 6 \
    --dim_feedforward 512 \
    --classifier_hidden 256 \
    --dropout 0.15 \
    --early_stopping 20 \
    --scheduler cosine \
    --warmup_epochs 5 \
    --noise_snr_min 10.0 \
    --noise_snr_max 25.0

echo ""
echo "============================================================"
echo "Entrenamiento completado"
echo "Resultados en: ./results_improved_v2/"
echo "============================================================"
echo ""
echo "Próximos pasos si aún no alcanzas 95%:"
echo "  1. Aumentar --train_subset a 100000 (doble de datos)"
echo "  2. Agregar features espectrales (FFT + magnitude)"
echo "  3. Implementar attention-based pooling en modelo"
echo "  4. Aumentar --warmup_epochs a 10"
echo "  5. Ajustar --noise_oversample_ratio si es necesario"
