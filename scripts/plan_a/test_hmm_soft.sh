#!/bin/bash

# Prueba hmm_soft con todos los K cacheados en ./cache/
# ETTh1, pred_len=96

echo "============================================================"
echo "PRUEBA: hmm_soft para cada K disponible - ETTh1 pred_len=96"
echo "============================================================"
echo ""

# Detectar Ks disponibles
K_VALUES=()
for f in ./cache/hmm_etth1_K*.pth; do
    k=$(echo "$f" | grep -oP 'K\K[0-9]+')
    K_VALUES+=($k)
done

# Ordenar
IFS=$'\n' K_VALUES=($(sort -n <<<"${K_VALUES[*]}")); unset IFS

echo "Ks disponibles: ${K_VALUES[@]}"
echo ""

for K in "${K_VALUES[@]}"; do
    echo "------------------------------------------------------------"
    echo "hmm_soft K=$K"
    echo "------------------------------------------------------------"

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
      --technique hmm_soft \
      --hmm_k $K \
      --des "Plan_A_hmm_soft_K${K}" \
      --itr 1

    if [ $? -eq 0 ]; then
        echo "[OK] K=$K completado"
    else
        echo "[ERROR] K=$K fallo"
    fi
    echo ""
done

echo "============================================================"
echo "RESUMEN - Revisa result_plan_a.txt para comparar MSE por K"
echo "============================================================"
grep "hmm_soft_K" result_plan_a.txt 2>/dev/null || echo "(sin resultados aun)"
