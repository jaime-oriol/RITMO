#!/bin/bash

# Script de prueba rápida del Plan A
# Ejecuta UN solo experimento para verificar que todo funciona
#
# Uso:
#   bash scripts/plan_a/test_plan_a.sh

echo "============================================================"
echo "PLAN A: PRUEBA RÁPIDA"
echo "============================================================"
echo ""
echo "Ejecutando experimento de prueba:"
echo "  - Técnica: patching (más simple para probar)"
echo "  - Dataset: ETTh1"
echo "  - Horizonte: 96"
echo "  - Epochs: 2 (para prueba rápida)"
echo ""

python -u run.py \
  --task_name plan_a \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_test \
  --model TransformerCommon \
  --data ETTh1 \
  --features S \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 128 \
  --n_heads 4 \
  --e_layers 2 \
  --d_ff 256 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 2 \
  --patience 3 \
  --technique patching \
  --des "Plan_A_test" \
  --itr 1

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ PRUEBA EXITOSA"
    echo "============================================================"
    echo ""
    echo "El pipeline del Plan A funciona correctamente."
    echo "Puedes ejecutar el script completo con:"
    echo "  bash scripts/plan_a/run_plan_a.sh"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "❌ PRUEBA FALLIDA"
    echo "============================================================"
    echo ""
    echo "Revisa los errores arriba para identificar el problema."
    echo ""
fi
