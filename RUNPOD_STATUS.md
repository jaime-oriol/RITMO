# RunPod Plan B — Status & Mañana

## SSH al pod

```bash
ssh root@74.15.1.150 -p 30593 -i ~/.ssh/id_ed25519
```

## Comandos rápidos para mañana

### 1. Ver estado de los jobs
```bash
ssh root@74.15.1.150 -p 30593 -i ~/.ssh/id_ed25519 'cd /workspace/RITMO && \
  echo "=== tmux ===" && tmux ls && \
  echo "=== Caches Electricity ===" && ls cache/hmm_M_electricity_K*.pth | wc -l && \
  echo "=== RITMO sweep results ===" && ls results/plan_b_PLANB_*_0 2>/dev/null | wc -l && \
  echo "=== SOTA results ===" && ls -d results/long_term_forecast_* 2>/dev/null | wc -l && \
  echo "=== Master log (últimas líneas) ===" && tail -20 logs/master.log'
```

**Esperado al terminar:**
- 6 caches Electricity (K=5..10)
- 256 results plan_b_PLANB
- 64 results long_term_forecast
- master.log con `=== DONE ritmo ===` y `=== DONE sota ===`

### 2. Si quieres ver los logs en directo
```bash
# Attach a la sesión RITMO
tmux attach -t ritmo
# Detach: Ctrl+B luego D

# Attach a la sesión SOTA
tmux attach -t sota
```

### 3. Cuando esté DONE, avísame y descargo todo

Yo correré (desde local):
```bash
scp -P 30593 -i ~/.ssh/id_ed25519 -r \
  root@74.15.1.150:/workspace/RITMO/results \
  root@74.15.1.150:/workspace/RITMO/cache \
  root@74.15.1.150:/workspace/RITMO/logs \
  /home/jaime/TFG/RITMO/_runpod_results/
```

### 4. Apagar el pod (parar de pagar)

En RunPod web:
- Click en el pod `ritmo-plan-b` → botón **Terminate** (rojo, definitivo)

---

## Plan en ejecución

### Phase 1 (ahora, ~30-60 min)
- HMM Electricity K=5..10 caches (6 workers paralelo, CPU)
- SOTA baselines 64 runs (GPU, paralelo desde inicio)

### Phase 2 (cuando HMM acabe, ~12-15h)
- RITMO sweep 256 runs (GPU)
- SOTA aún corriendo si no ha acabado (comparte GPU sin problema)

### Total esperado
- ~13-15h wall-clock
- ~$5 coste

---

## Si algo va mal

**Re-attach y ver logs:**
```bash
tmux attach -t ritmo  # Ctrl+B D para salir
tmux attach -t sota
```

**Revisar logs persistentes:**
```bash
tail -50 /workspace/RITMO/logs/hmm.log
tail -50 /workspace/RITMO/logs/ritmo.log
tail -50 /workspace/RITMO/logs/sota.log
tail -20 /workspace/RITMO/logs/master.log
```

**Re-lanzar (resumible — salta lo hecho):**
```bash
cd /workspace/RITMO && bash launch.sh
```
