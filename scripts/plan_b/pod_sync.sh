#!/usr/bin/env bash
# Sincroniza resultados del pod a local cada 3 minutos.
# Uso: bash scripts/plan_b/pod_sync.sh <host> <port>

HOST=$1
PORT=$2
LOCAL_DEST="./_runpod_results/plan_b_sweep/"

if [[ -z "$HOST" || -z "$PORT" ]]; then
  echo "Uso: bash scripts/plan_b/pod_sync.sh <host> <port>"
  exit 1
fi

echo "[sync] Arrancando sync cada 3 min → $LOCAL_DEST"
mkdir -p "$LOCAL_DEST"

while true; do
  TS=$(date '+%H:%M:%S')
  # Resultados ya movidos a _runpod_results/
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
    "root@${HOST}:/workspace/RITMO/_runpod_results/plan_b_sweep/" \
    "$LOCAL_DEST" 2>/dev/null
  # Resultados frescos en results/ (por si el pod muere antes de moverlos)
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
    "root@${HOST}:/workspace/RITMO/results/" \
    "$LOCAL_DEST" 2>/dev/null

  DONE=$(find "$LOCAL_DEST" -name "metrics.npy" 2>/dev/null | wc -l)
  echo "[$TS] sync OK — $DONE/256 completados en local"
  sleep 180
done
