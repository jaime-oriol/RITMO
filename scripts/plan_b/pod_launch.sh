#!/usr/bin/env bash
# Setup completo + lanzamiento sweep Plan B completo en RunPod.
# Corre los 102 runs faltantes (1 ETTh2 + 37 Weather + 64 Electricity).
# Ejecutar desde la raiz del repo LOCAL:
#   bash scripts/plan_b/pod_launch.sh <host> <port>

set -e

HOST=$1
PORT=$2

if [[ -z "$HOST" || -z "$PORT" ]]; then
  echo "Uso: bash scripts/plan_b/pod_launch.sh <host> <port>"
  exit 1
fi

SSH="ssh -o StrictHostKeyChecking=no -p $PORT root@$HOST"
SCP="scp -o StrictHostKeyChecking=no -P $PORT -r"

echo "=== 1. Verificar GPU y CUDA ==="
$SSH "nvidia-smi | head -12"

echo ""
echo "=== 2. Clonar repo ==="
$SSH "git clone https://github.com/jaime-oriol/RITMO.git /workspace/RITMO 2>/dev/null \
  || (cd /workspace/RITMO && git pull --ff-only)"

echo ""
echo "=== 3. Instalar dependencias ==="
$SSH "pip install -q pandas scikit-learn einops numba tqdm reformer-pytorch && echo 'pip OK'"
# Verificar que torch+CUDA sigue intacto (reformer-pytorch puede sobreescribirlo)
$SSH "python -c \"import torch; assert torch.cuda.is_available(), 'CUDA PERDIDA'; \
  print('torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))\""

echo ""
echo "=== 4. Crear directorios ==="
$SSH "mkdir -p /workspace/RITMO/cache \
               /workspace/RITMO/logs \
               /workspace/RITMO/dataset/ETT-small \
               /workspace/RITMO/dataset/weather \
               /workspace/RITMO/dataset/electricity"

echo ""
echo "=== 5. Subir datasets (ETT-small 24M, weather 7M, electricity 92M) ==="
scp -o StrictHostKeyChecking=no -P $PORT \
  dataset/ETT-small/ETTh1.csv dataset/ETT-small/ETTh2.csv \
  "root@${HOST}:/workspace/RITMO/dataset/ETT-small/"
scp -o StrictHostKeyChecking=no -P $PORT \
  dataset/weather/weather.csv \
  "root@${HOST}:/workspace/RITMO/dataset/weather/"
scp -o StrictHostKeyChecking=no -P $PORT \
  dataset/electricity/electricity.csv \
  "root@${HOST}:/workspace/RITMO/dataset/electricity/"
echo "  datasets OK"

echo ""
echo "=== 6. Subir 32 caches HMM-M ==="
for ds in etth1 etth2 weather electricity; do
  for k in 3 4 5 6 7 8 9 10; do
    FILE="hmm_M_${ds}_K${k}.pth"
    if [[ -f "cache/$FILE" ]]; then
      SRC="cache/$FILE"
    elif [[ -f "_runpod_results/caches_hmm_M/$FILE" ]]; then
      SRC="_runpod_results/caches_hmm_M/$FILE"
    else
      echo "  WARN: falta $FILE"
      continue
    fi
    scp -o StrictHostKeyChecking=no -P $PORT "$SRC" \
      "root@${HOST}:/workspace/RITMO/cache/$FILE"
  done
done
echo "  caches OK"

echo ""
echo "=== 7. Subir marcadores de runs ya completados (evita re-correrlos) ==="
# Empaquetar solo los metrics.npy de resultados ya guardados (~18 KB)
tar czf /tmp/done_markers.tar.gz \
  $(find _runpod_results/plan_b_sweep/ -name "metrics.npy" 2>/dev/null)
scp -o StrictHostKeyChecking=no -P $PORT \
  /tmp/done_markers.tar.gz "root@${HOST}:/tmp/"
$SSH "cd /workspace/RITMO && tar xzf /tmp/done_markers.tar.gz && \
  echo \"  marcadores extraidos: \$(find _runpod_results/plan_b_sweep/ -name metrics.npy 2>/dev/null | wc -l) resultados ya hechos\""

echo ""
echo "=== 8. Verificar setup ==="
$SSH "cd /workspace/RITMO && python -c \"
import sys; sys.path.insert(0, '.')
from pathlib import Path
from scripts.plan_b.complete_missing import cache_path, result_exists
from scripts.plan_b.plan_b_config import DATASETS, K_VALUES_BY_DATASET, VARIANTS, HORIZONS, experiment_tag
import torch

# Caches
missing_caches = [(ds['name'], k) for ds in DATASETS
                  for k in K_VALUES_BY_DATASET[ds['name']]
                  if not Path(cache_path(ds['name'], k)).exists()]
print('Caches faltantes:', missing_caches if missing_caches else 'NINGUNA')

# Runs pendientes
pending = [(ds['name'], v, k, pl) for ds in DATASETS
           for k in K_VALUES_BY_DATASET[ds['name']]
           for v in VARIANTS for pl in HORIZONS
           if not result_exists(experiment_tag(ds['name'], v, k, pl))]
from collections import Counter
by_ds = Counter(p[0] for p in pending)
print('Runs pendientes:', len(pending), dict(by_ds))

# GPU
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
\""

echo ""
echo "=== 9. Lanzar 2 shards paralelos (todos los datasets) ==="
$SSH "cd /workspace/RITMO && nohup python -u scripts/plan_b/complete_missing.py \
  --use-gpu 1 --num-workers 6 --shard 0/2 \
  > logs/sweep_s0.log 2>&1 & echo \"shard 0 PID=\$!\""

sleep 3

$SSH "cd /workspace/RITMO && nohup python -u scripts/plan_b/complete_missing.py \
  --use-gpu 1 --num-workers 6 --shard 1/2 \
  > logs/sweep_s1.log 2>&1 & echo \"shard 1 PID=\$!\""

echo ""
echo "========================================"
echo "Setup completo. 102 runs lanzados."
echo ""
echo "Monitorear:"
echo "  $SSH 'tail -f /workspace/RITMO/logs/sweep_s0.log'"
echo "  $SSH 'tail -f /workspace/RITMO/logs/sweep_s1.log'"
echo ""
echo "Sincronizar a local cada 3 min (otra terminal):"
echo "  bash scripts/plan_b/pod_sync.sh $HOST $PORT"
echo ""
echo "GPU utilization:"
echo "  $SSH 'watch -n5 nvidia-smi'"
echo "========================================"
