"""Corre los runs faltantes del Plan B y los guarda en _runpod_results/plan_b_sweep/.

Resumible: salta runs que ya existen en _runpod_results/plan_b_sweep/ O en results/.
Caches: busca en ./cache/ (standard) con fallback a _runpod_results/caches_hmm_M/.

Uso:
    python -u scripts/plan_b/complete_missing.py
    python -u scripts/plan_b/complete_missing.py --only-dataset Electricity
    python -u scripts/plan_b/complete_missing.py --use-gpu 1 --num-workers 8 --shard 0/2
"""
import argparse
import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from scripts.plan_b.plan_b_config import (
    DATASETS, K_VALUES_BY_DATASET, VARIANTS, HORIZONS, experiment_tag,
    cache_path as _config_cache_path,
)

DEST_DIR = REPO_ROOT / '_runpod_results' / 'plan_b_sweep'

# Fallback: si la cache no está en ./cache/, buscar en _runpod_results/caches_hmm_M/
_LEGACY_CACHE_DIR = REPO_ROOT / '_runpod_results' / 'caches_hmm_M'


def cache_path(dataset_name: str, K: int) -> str:
    """Devuelve la ruta de la cache: ./cache/ primero, _runpod_results/ como fallback."""
    canonical = Path(_config_cache_path(dataset_name, K))
    if canonical.exists():
        return str(canonical)
    legacy = _LEGACY_CACHE_DIR / f'hmm_M_{dataset_name.lower()}_K{K}.pth'
    if legacy.exists():
        return str(legacy)
    return str(canonical)  # devuelve canonical aunque no exista (produce error legible)


def result_exists(tag: str) -> bool:
    """True si existe metrics.npy en DEST_DIR o en results/ (cubre restart tras crash)."""
    for base in (DEST_DIR, Path('results')):
        for p in base.glob(f'plan_b_{tag}_*_{tag}_0'):
            if (p / 'metrics.npy').exists():
                return True
    return False


def find_local_result(tag: str) -> Path | None:
    for p in Path('results').glob(f'plan_b_{tag}_*_{tag}_0'):
        if (p / 'metrics.npy').exists():
            return p
    return None


def build_cmd(ds, variant, K, pred_len, num_workers, use_gpu):
    tag = experiment_tag(ds['name'], variant, K, pred_len)
    return tag, [
        sys.executable, '-u', 'run.py',
        '--task_name', 'plan_b', '--is_training', '1',
        '--root_path', ds['root'], '--data_path', ds['data_path'],
        '--model_id', tag, '--model', 'TransformerCommon', '--data', ds['data_arg'],
        '--features', 'M', '--target', 'OT', '--freq', ds['freq'],
        '--seq_len', '96', '--label_len', '48', '--pred_len', str(pred_len),
        '--enc_in', str(ds['C']), '--dec_in', str(ds['C']), '--c_out', str(ds['C']),
        '--d_model', '64', '--n_heads', '4', '--e_layers', '2', '--d_ff', '128',
        '--dropout', '0.1',
        '--batch_size', str(ds['batch_size']),
        '--learning_rate', '0.001', '--lradj', 'cosine',
        '--train_epochs', str(ds['train_epochs']), '--patience', '7',
        '--use_gpu', str(use_gpu), '--num_workers', str(num_workers),
        '--technique', variant, '--hmm_k', str(K),
        '--hmm_cache_path', cache_path(ds['name'], K),
        '--des', tag, '--itr', '1',
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only-dataset', default=None)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--use-gpu', type=int, default=0)
    ap.add_argument('--shard', type=str, default='0/1',
                    help='X/N: procesa solo tasks con idx %% N == X.')
    args = ap.parse_args()

    shard_x, shard_n = (int(x) for x in args.shard.split('/'))

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)

    datasets = DATASETS
    if args.only_dataset:
        datasets = [d for d in DATASETS if d['name'] == args.only_dataset]

    env = os.environ.copy()
    env.update({'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})

    runs = [
        (ds, variant, K, pl)
        for ds in datasets
        for K in K_VALUES_BY_DATASET[ds['name']]
        for variant in VARIANTS
        for pl in HORIZONS
    ]
    runs = [r for i, r in enumerate(runs) if i % shard_n == shard_x]

    pending = [(ds, v, K, pl) for ds, v, K, pl in runs
               if not result_exists(experiment_tag(ds['name'], v, K, pl))]
    print(
        f"[complete:s{shard_x}/{shard_n}] total={len(runs)}  "
        f"pendientes={len(pending)}  ya_hechos={len(runs)-len(pending)}",
        flush=True,
    )

    done = skipped = failed = 0
    t_all = time.time()

    for ds, variant, K, pred_len in runs:
        tag = experiment_tag(ds['name'], variant, K, pred_len)

        if result_exists(tag):
            skipped += 1
            continue

        cp = cache_path(ds['name'], K)
        if not Path(cp).exists():
            print(f"[complete] SKIP sin_cache {tag}  ({cp})", flush=True)
            failed += 1
            continue

        print(f"\n[complete] RUN {tag}", flush=True)
        t0 = time.time()
        _, cmd = build_cmd(ds, variant, K, pred_len, args.num_workers, args.use_gpu)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        dt = time.time() - t0

        if proc.returncode != 0:
            failed += 1
            err = '\n'.join(proc.stderr.strip().splitlines()[-12:])
            print(f"[complete] FAIL {tag} ({dt:.1f}s)\n{err}", flush=True)
        else:
            local = find_local_result(tag)
            if local:
                dest = DEST_DIR / local.name
                shutil.move(str(local), str(dest))
                # Borrar arrays grandes — solo metrics.npy es necesario para el análisis
                for fname in ('pred.npy', 'true.npy'):
                    (dest / fname).unlink(missing_ok=True)
                done += 1
                metric_lines = [ln for ln in proc.stdout.splitlines() if 'mse:' in ln]
                msg = metric_lines[-1] if metric_lines else ''
                print(f"[complete] OK {tag} ({dt:.1f}s)  {msg}", flush=True)
            else:
                failed += 1
                print(f"[complete] FAIL {tag} — result dir no encontrado tras run OK", flush=True)

        gc.collect()

    print(
        f"\n[complete] Fin. done={done} skipped={skipped} failed={failed} "
        f"wall={time.time()-t_all:.1f}s",
        flush=True,
    )


if __name__ == '__main__':
    main()
