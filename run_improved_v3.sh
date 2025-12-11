#!/bin/bash

# ENTRENAMIENTO MEJORADO CON TODOS LOS FIXES
# Basado en análisis profundo del por qué no se alcanza 95%

echo "============================================================"
echo "ENTRENAMIENTO OPTIMIZADO - Meta: 230k parámetros + P-wave"
echo "============================================================"
echo ""
echo "Mejoras aplicadas:"
echo "  ✅ Normalización Z-score correcta (mean - mean, divide por std)"
echo "  ✅ P-wave alignment (¡SOLO P-wave, no S-wave!)"
echo "  ✅ Modelo eficiente (~230k params: d_model=96, 4 layers)"
echo "  ✅ SNR realista (10-25 dB para augmentation)"
echo "  ✅ Balanced augmentation (noise class only)"
echo "  ✅ Cosine scheduler con warmup"
echo ""
echo "Configuración:"
echo "  • Parámetros: ~230,000 (meta original)"
echo "  • Epochs: 30 (rápido para experimentar)"
echo "  • Phase: P-wave only (máxima discriminación)"
echo "  • Normalización: Z-score + clip [-3, 3]"
echo "  • P-wave alignment: window centrada en P-wave"
echo ""
echo "Impacto esperado:"
echo "  • P-wave alignment: +10-15%"
echo "  • Normalización correcta: +5-8%"
echo "  • Total: +15-23%"
echo "  • Meta anterior: 59%"
echo "  • Esperado: 74-82%"
echo ""
echo "============================================================"
echo ""

python run_experiment.py \
    --lazy_load \
    --region latam \
    --phase P \
    --augment \
    --balanced_aug \
    --epochs 30 \
    --batch 64 \
    --save_dir ./results_improved_v3 \
    --threshold 0.5 \
    --loss_type bce \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --d_model 96 \
    --num_layers 4 \
    --dim_feedforward 384 \
    --classifier_hidden 200 \
    --dropout 0.15 \
    --early_stopping 15 \
    --scheduler cosine \
    --warmup_epochs 3 \
    --noise_snr_min 10.0 \
    --noise_snr_max 25.0

echo ""
echo "============================================================"
echo "Entrenamiento completado"
echo "Resultados en: ./results_improved_v3/"
echo "============================================================"
echo ""
echo "Siguiente: Revisar resultados en epoch 30"
echo "Si accuracy < 75%: aumentar --d_model a 112 o --num_layers a 5"
echo "Si accuracy > 82%: reducir params o aumentar epochs a 50"
echo ""
