#!/bin/bash

# Script de ejecución completa del Plan A
# Ejecuta comparación controlada de 6 técnicas de tokenización
# en 6 datasets con 4 horizontes de predicción
#
# Total: 6 técnicas × 6 datasets × 4 horizontes = 144 experimentos
#
# Uso:
#   bash scripts/plan_a/run_plan_a.sh

# === CONFIGURACIÓN ===

# Técnicas de tokenización a comparar
TECHNIQUES=("discretization" "text_based" "patching" "decomposition" "foundation" "hmm")

# Datasets para evaluación
DATASETS=("ETTh1" "ETTh2" "Weather" "Electricity" "Traffic" "Exchange")

# Horizontes de predicción
PRED_LENS=(96 192 336 720)

# Hiperparámetros FIJOS para Plan A (comparación justa)
# Estos valores son iguales para todas las técnicas para asegurar comparación controlada
SEQ_LEN=96           # Longitud de secuencia de entrada
D_MODEL=256          # Dimensión de embeddings del Transformer (capacidad del modelo)
N_HEADS=4            # Número de attention heads (paralelización de atención)
E_LAYERS=3           # Número de capas del encoder (profundidad del modelo)
D_FF=512             # Dimensión de feed-forward (2×d_model, estándar en Transformers)
DROPOUT=0.1          # Tasa de dropout para regularización (evita overfitting)
BATCH_SIZE=32        # Tamaño de batch (compromiso memoria/estabilidad)
LEARNING_RATE=0.0001 # Learning rate del optimizador (controla tamaño de paso en gradiente)
TRAIN_EPOCHS=20      # Máximo de epochs de entrenamiento
PATIENCE=5           # Epochs sin mejora antes de early stopping

# === FUNCIONES AUXILIARES ===

get_root_path() {
    # Obtiene root_path según dataset
    case $1 in
        ETTh1|ETTh2)
            echo "./dataset/ETT-small/"
            ;;
        Weather)
            echo "./dataset/weather/"
            ;;
        Electricity)
            echo "./dataset/electricity/"
            ;;
        Traffic)
            echo "./dataset/traffic/"
            ;;
        Exchange)
            echo "./dataset/exchange_rate/"
            ;;
        *)
            echo "./dataset/"
            ;;
    esac
}

get_data_path() {
    # Obtiene data_path según dataset
    case $1 in
        ETTh1)
            echo "ETTh1.csv"
            ;;
        ETTh2)
            echo "ETTh2.csv"
            ;;
        Weather)
            echo "weather.csv"
            ;;
        Electricity)
            echo "electricity.csv"
            ;;
        Traffic)
            echo "traffic.csv"
            ;;
        Exchange)
            echo "exchange_rate.csv"
            ;;
        *)
            echo "data.csv"
            ;;
    esac
}

get_enc_in() {
    # Obtiene enc_in según dataset (para univariado S, siempre 1)
    # Para multivariado M:
    case $1 in
        ETTh1|ETTh2)
            echo 7
            ;;
        Weather)
            echo 21
            ;;
        Electricity)
            echo 321
            ;;
        Traffic)
            echo 862
            ;;
        Exchange)
            echo 8
            ;;
        *)
            echo 1
            ;;
    esac
}

check_experiment_completed() {
    # Verifica si un experimento ya fue completado
    # Args: technique, dataset, pred_len
    local technique=$1
    local dataset=$2
    local pred_len=$3

    # Patrón del directorio de resultados (más flexible)
    # Formato: ./results/plan_a_{dataset}_{seq_len}_{pred_len}_*
    local result_pattern="./results/plan_a_${dataset}_${SEQ_LEN}_${pred_len}_*"

    # Comprobar si existe el directorio y contiene metrics.npy
    if compgen -G "$result_pattern" > /dev/null; then
        # Verificar que contiene el archivo metrics.npy (señal de que completó)
        for dir in $result_pattern; do
            # Comprobar que el directorio corresponde a la técnica (case-insensitive)
            if echo "$dir" | grep -iq "Plan_A.*${technique}"; then
                if [ -f "$dir/metrics.npy" ]; then
                    return 0  # Completado
                fi
            fi
        done
    fi

    return 1  # No completado
}

# === PIPELINE PRINCIPAL ===

echo "============================================================"
echo "PLAN A: COMPARACIÓN CONTROLADA DE TÉCNICAS DE TOKENIZACIÓN"
echo "============================================================"
echo ""
echo "Configuración:"
echo "  - Técnicas: ${TECHNIQUES[@]}"
echo "  - Datasets: ${DATASETS[@]}"
echo "  - Horizontes: ${PRED_LENS[@]}"
echo "  - Total experimentos: $((${#TECHNIQUES[@]} * ${#DATASETS[@]} * ${#PRED_LENS[@]}))"
echo ""
echo "Hiperparámetros fijos:"
echo "  - seq_len: $SEQ_LEN"
echo "  - d_model: $D_MODEL"
echo "  - n_heads: $N_HEADS"
echo "  - e_layers: $E_LAYERS"
echo "  - d_ff: $D_FF"
echo "  - dropout: $DROPOUT"
echo "  - batch_size: $BATCH_SIZE"
echo "  - learning_rate: $LEARNING_RATE"
echo "  - train_epochs: $TRAIN_EPOCHS"
echo "  - patience: $PATIENCE"
echo ""
echo "============================================================"
echo ""

# Contadores de experimentos
counter=0
completed=0
skipped=0
total_experiments=$((${#TECHNIQUES[@]} * ${#DATASETS[@]} * ${#PRED_LENS[@]}))

echo "Verificando experimentos ya completados..."
echo ""

# Triple loop: Técnica × Dataset × Horizonte
for technique in "${TECHNIQUES[@]}"; do
    echo ""
    echo "=========================================="
    echo "TÉCNICA: $technique"
    echo "=========================================="
    echo ""

    for data in "${DATASETS[@]}"; do
        echo ""
        echo ">>> Dataset: $data"
        echo ""

        # Obtener paths y configuración del dataset
        root_path=$(get_root_path $data)
        data_path=$(get_data_path $data)
        enc_in=$(get_enc_in $data)

        for pred_len in "${PRED_LENS[@]}"; do
            counter=$((counter + 1))

            echo "----------------------------"
            echo "Experimento $counter/$total_experiments"
            echo "Técnica: $technique"
            echo "Dataset: $data"
            echo "Horizonte: $pred_len"
            echo "----------------------------"

            # === VERIFICAR SI YA COMPLETADO ===
            if check_experiment_completed "$technique" "$data" "$pred_len"; then
                echo "[SKIP] Ya completado"
                echo ""
                skipped=$((skipped + 1))
                continue
            fi

            # Identificador del experimento
            model_id="${data}_${SEQ_LEN}_${pred_len}"

            # === EJECUTAR EXPERIMENTO ===
            python -u run.py \
              --task_name plan_a \
              --is_training 1 \
              --root_path $root_path \
              --data_path $data_path \
              --model_id $model_id \
              --model TransformerCommon \
              --data $data \
              --features S \
              --seq_len $SEQ_LEN \
              --pred_len $pred_len \
              --enc_in 1 \
              --dec_in 1 \
              --c_out 1 \
              --d_model $D_MODEL \
              --n_heads $N_HEADS \
              --e_layers $E_LAYERS \
              --d_ff $D_FF \
              --dropout $DROPOUT \
              --batch_size $BATCH_SIZE \
              --learning_rate $LEARNING_RATE \
              --train_epochs $TRAIN_EPOCHS \
              --patience $PATIENCE \
              --use_gpu 0 \
              --technique $technique \
              --des "Plan_A_${technique}" \
              --itr 1

            if [ $? -eq 0 ]; then
                echo ""
                echo "[OK] Completado: $technique - $data - pred_len=$pred_len"
                echo ""
                completed=$((completed + 1))
            else
                echo ""
                echo "[ERROR] $technique - $data - pred_len=$pred_len"
                echo "El script continuará con el siguiente experimento..."
                echo ""
            fi
        done
    done
done

echo ""
echo "============================================================"
echo "PLAN A COMPLETADO"
echo "============================================================"
echo ""
echo "Estadísticas:"
echo "  - Total experimentos: $total_experiments"
echo "  - Ejecutados en esta sesión: $completed"
echo "  - Saltados (ya completados): $skipped"
echo "  - Progreso total: $((completed + skipped))/$total_experiments ($((100 * (completed + skipped) / total_experiments))%)"
echo ""
echo "Resultados guardados en:"
echo "  - Checkpoints: ./checkpoints/"
echo "  - Test results: ./test_results/"
echo "  - Métricas: ./results/"
echo "  - Resumen: result_plan_a.txt"
echo ""
echo "============================================================"
