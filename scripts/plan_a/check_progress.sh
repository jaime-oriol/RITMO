#!/bin/bash

# Script para verificar progreso del Plan A sin ejecutar experimentos
# Muestra cuántos experimentos están completados y cuáles faltan

# === CONFIGURACIÓN (debe coincidir con run_plan_a.sh) ===
TECHNIQUES=("discretization" "text_based" "patching" "decomposition" "foundation" "hmm")
DATASETS=("ETTh1" "ETTh2" "Weather" "Electricity" "Traffic" "Exchange")
PRED_LENS=(96 192 336 720)
SEQ_LEN=96

# === FUNCIÓN DE VERIFICACIÓN ===
check_experiment_completed() {
check_experiment_completed() {
    local technique=$1
    local dataset=$2
    local pred_len=$3

    local result_pattern="./results/plan_a_${dataset}_${SEQ_LEN}_${pred_len}_*"

    if compgen -G "$result_pattern" > /dev/null; then
        for dir in $result_pattern; do
            if echo "$dir" | grep -iq "Plan_A.*${technique}"; then
                if [ -f "$dir/metrics.npy" ]; then
                    return 0
                fi
            fi
        done
    fi

    return 1
}
    local dataset=$2
    local pred_len=$3

    local result_pattern="./results/plan_a_${dataset}_${SEQ_LEN}_${pred_len}_*"

    if compgen -G "$result_pattern" > /dev/null; then
        for dir in $result_pattern; do
            if echo "$dir" | grep -iq "Plan_A.*${technique}"; then
            if [ -f "$dir/metrics.npy" ]; then
                return 0  # Completado
                fi
            fi
        done
    fi

    return 1  # No completado
}

# === ANÁLISIS DE PROGRESO ===

echo "============================================================"
echo "PROGRESO DEL PLAN A"
echo "============================================================"
echo ""

total=0
completed=0
pending=0

declare -A technique_counts
declare -A dataset_counts

for technique in "${TECHNIQUES[@]}"; do
    technique_counts[$technique]=0
done

for dataset in "${DATASETS[@]}"; do
    dataset_counts[$dataset]=0
done

# Lista de experimentos pendientes
pending_list=()

for technique in "${TECHNIQUES[@]}"; do
    for data in "${DATASETS[@]}"; do
        for pred_len in "${PRED_LENS[@]}"; do
            total=$((total + 1))

            if check_experiment_completed "$technique" "$data" "$pred_len"; then
                completed=$((completed + 1))
                technique_counts[$technique]=$((${technique_counts[$technique]} + 1))
                dataset_counts[$data]=$((${dataset_counts[$data]} + 1))
            else
                pending=$((pending + 1))
                pending_list+=("$technique | $data | pred_len=$pred_len")
            fi
        done
    done
done

# === MOSTRAR ESTADÍSTICAS ===

echo "Progreso General:"
echo "  [OK] Completados: $completed/$total ($((100 * completed / total))%)"
echo "  ⏳ Pendientes: $pending/$total ($((100 * pending / total))%)"
echo ""

echo "Progreso por Técnica:"
for technique in "${TECHNIQUES[@]}"; do
    count=${technique_counts[$technique]}
    total_per_technique=$((${#DATASETS[@]} * ${#PRED_LENS[@]}))
    percentage=$((100 * count / total_per_technique))
    printf "  %-15s %2d/%-2d (%3d%%) " "$technique" $count $total_per_technique $percentage
    # Barra de progreso
    completed_bars=$((percentage / 5))
    remaining_bars=$((20 - completed_bars))
    printf "["
    for ((i=0; i<completed_bars; i++)); do printf "="; done
    for ((i=0; i<remaining_bars; i++)); do printf " "; done
    printf "]\n"
done
echo ""

echo "Progreso por Dataset:"
for data in "${DATASETS[@]}"; do
    count=${dataset_counts[$data]}
    total_per_dataset=$((${#TECHNIQUES[@]} * ${#PRED_LENS[@]}))
    percentage=$((100 * count / total_per_dataset))
    printf "  %-12s %2d/%-2d (%3d%%) " "$data" $count $total_per_dataset $percentage
    completed_bars=$((percentage / 5))
    remaining_bars=$((20 - completed_bars))
    printf "["
    for ((i=0; i<completed_bars; i++)); do printf "="; done
    for ((i=0; i<remaining_bars; i++)); do printf " "; done
    printf "]\n"
done
echo ""

if [ $pending -gt 0 ]; then
    echo "Experimentos Pendientes:"
    for exp in "${pending_list[@]}"; do
        echo "  - $exp"
    done
    echo ""
fi

echo "============================================================"
echo ""

if [ $pending -eq 0 ]; then
    echo " PLAN A COMPLETADO AL 100%"
else
    echo "Para continuar: bash scripts/plan_a/run_plan_a.sh"
fi

echo ""
