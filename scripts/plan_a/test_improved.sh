#!/bin/bash

# Script de prueba con hiperparámetros mejorados
# Ejecuta UN SOLO experimento (hmm, ETTh1, pred_len=96) para verificar
# que las mejoras funcionan antes de ejecutar los 144 experimentos

echo "============================================================"
echo "PRUEBA DE HIPERPARÁMETROS MEJORADOS"
echo "============================================================"
echo ""
echo "Experimento: HMM × ETTh1 × pred_len=96"
echo ""
echo "Hiperparámetros MEJORADOS:"
echo "  - learning_rate: 0.0001 (antes: 0.001)"
echo "  - patience: 5 (antes: 3)"
echo "  - train_epochs: 20 (antes: 10)"
echo "  - d_model: 256 (antes: 128)"
echo "  - e_layers: 3 (antes: 2)"
echo "  - d_ff: 512 (antes: 256)"
echo ""
echo "Visualizaciones MEJORADAS:"
echo "  - Más timesteps (input + output completo)"
echo "  - Datos desnormalizados"
echo "  - MSE y MAE en título"
echo ""
echo "============================================================"
echo ""

# Limpiar experimento anterior si existe
echo "Limpiando experimento anterior (hmm, ETTh1, 96)..."
rm -rf ./results/plan_a_ETTh1_96_96_*_Plan_A_*HMM* 2>/dev/null
rm -rf ./checkpoints/plan_a_ETTh1_96_96_*_Plan_A_*HMM* 2>/dev/null
rm -rf ./test_results/plan_a_ETTh1_96_96_*_Plan_A_*HMM* 2>/dev/null
echo "[OK] Limpiado"
echo ""

echo "Ejecutando experimento de prueba..."
echo ""

python -u run.py \
  --task_name plan_a \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model TransformerCommon \
  --data ETTh1 \
  --features S \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 256 \
  --n_heads 4 \
  --e_layers 3 \
  --d_ff 512 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 20 \
  --patience 5 \
  --use_gpu 0 \
  --technique hmm \
  --des "Plan_A_test_improved" \
  --itr 1

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "[OK] PRUEBA COMPLETADA"
    echo "============================================================"
    echo ""
    echo "Revisa los resultados:"
    echo "  - Métricas: result_plan_a.txt (última entrada)"
    echo "  - Gráficos: ./test_results/plan_a_ETTh1_96_96_*/"
    echo ""
    echo "Si las predicciones se ven BIEN (no planas):"
    echo "  → Ejecuta: bash scripts/plan_a/run_plan_a.sh"
    echo ""
    echo "Si todavía se ven MAL:"
    echo "  → Revisar y ajustar más hiperparámetros"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "[ERROR] ERROR EN LA PRUEBA"
    echo "============================================================"
    echo ""
    echo "Revisa los logs arriba para ver qué falló."
    echo ""
fi
