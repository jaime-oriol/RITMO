"""Entrena las caches HMM-M (compartidas entre canales) para Plan B.

Resumible: si existe la cache en disco, la salta.
Todas las log-lines van a stdout (apto para `tee` o `nohup ... >`).

Uso:
    python -u scripts/plan_b/train_hmm_caches.py [--datasets ETTh1 ETTh2 ...]
    python -u scripts/plan_b/train_hmm_caches.py --only-dataset Electricity --ks 3

Diseño cientifico:
    - Entrenamiento sobre train split (sin leakage).
    - Normalizacion z-score per-canal con estadisticas de train.
    - K-means init reproducible (seed 2021).
    - Apila C canales como B secuencias independientes -> baum_welch_batch.
    - Memory-aware via chunk_size por dataset (ECL chunk=8 para no reventar RAM).
"""
import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Raiz del proyecto
REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from scripts.plan_b.plan_b_config import (  # noqa: E402
    DATASETS, K_VALUES_BY_DATASET, SEED, cache_path,
)
from hmm import baum_welch_batch  # noqa: E402
from hmm.checkpoint import save_hmm_params  # noqa: E402

# Limitar threads (WSL-safe, evita oversubscription con numpy/scipy)
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

# Fijar semillas
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def load_train_multichannel(ds):
    """Lee CSV, descarta 'date', aplica z-score per-canal con stats de train split.

    Returns:
        obs: [C, T_train] normalizado, alineado con el orden que entregara el data_loader.
    """
    df = pd.read_csv(ds['csv'])
    cols = [c for c in df.columns if c != 'date']
    # Nota: Dataset_Custom reordena target al final pero OT ya es la ultima columna
    # en Weather y ECL, por lo que el orden resultante es identico.
    data = df[cols].values.astype(np.float64)
    T = data.shape[0]
    if ds['split_type'] == 'ETT':
        train_end = 12 * 30 * 24
    else:
        train_end = int(T * 0.7)
    train = data[:train_end]
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-8
    train_norm = (train - mu) / sd
    return train_norm.T.copy(), train_end, T


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=None,
                    help='Lista de nombres (ETTh1 ETTh2 Weather Electricity). Default: todos.')
    ap.add_argument('--ks', nargs='*', type=int, default=None,
                    help='Override manual de K values (aplicado a todos los datasets).')
    ap.add_argument('--skip-electricity', action='store_true',
                    help='Salta Electricity (util si el sweep completo es inviable en CPU).')
    ap.add_argument('--workers', type=int, default=1,
                    help='Procesos paralelos por (K) dentro de cada dataset. Default 1 (secuencial).')
    return ap.parse_args()


def _train_one_K(args_tuple):
    """Worker para multiprocessing.Pool. Entrena un (dataset, K) y guarda en disco."""
    name, obs, K, chunk_size, cp = args_tuple
    if Path(cp).exists():
        return (K, 'SKIP', None, 0.0)
    t_k = time.time()
    params = baum_welch_batch(
        obs, K=K, max_iter=2000, epsilon=1e-4,
        random_state=SEED, chunk_size=chunk_size,
        verbose=False,
    )
    save_hmm_params(params, cp)
    dt = time.time() - t_k
    return (K, 'OK', {
        'll': float(params['log_likelihood']),
        'iters': int(params['n_iter']),
        'converged': bool(params['converged']),
    }, dt)


def main():
    args = parse_args()
    Path('cache').mkdir(exist_ok=True)

    datasets = DATASETS
    if args.datasets:
        datasets = [d for d in DATASETS if d['name'] in set(args.datasets)]
    if args.skip_electricity:
        datasets = [d for d in datasets if d['name'] != 'Electricity']

    total_start = time.time()
    print(f"[plan_b-caches] Inicio. datasets={[d['name'] for d in datasets]} workers={args.workers}", flush=True)

    for ds in datasets:
        name = ds['name']
        Ks = args.ks if args.ks else K_VALUES_BY_DATASET[name]
        print(f"\n[plan_b-caches] === {name} (C={ds['C']}) Ks={Ks} chunk={ds['hmm_chunk_size']} ===",
              flush=True)

        # Decidir si todas las caches ya existen para evitar cargar el CSV.
        missing = [K for K in Ks if not Path(cache_path(name, K)).exists()]
        if not missing:
            print(f"[plan_b-caches] {name}: todas las caches ya existen. SKIP.", flush=True)
            continue

        t0 = time.time()
        obs, train_end, T = load_train_multichannel(ds)
        print(f"[plan_b-caches] {name}: obs.shape={obs.shape} train_end={train_end} "
              f"T_total={T} mean={obs.mean():.4f} std={obs.std():.4f} "
              f"({time.time()-t0:.1f}s loading)", flush=True)

        # Lista de tareas (solo Ks que faltan)
        tasks = [(name, obs, K, ds['hmm_chunk_size'], cache_path(name, K))
                 for K in Ks if not Path(cache_path(name, K)).exists()]
        for K in Ks:
            if Path(cache_path(name, K)).exists():
                print(f"  SKIP {cache_path(name, K)} (exists)", flush=True)

        if args.workers > 1 and len(tasks) > 1:
            from multiprocessing import Pool
            n = min(args.workers, len(tasks))
            print(f"  Lanzando {len(tasks)} entrenamientos en {n} procesos paralelos...", flush=True)
            with Pool(processes=n) as pool:
                for K, status, info, dt in pool.imap_unordered(_train_one_K, tasks):
                    if status == 'OK':
                        print(f"    K={K} OK  LL={info['ll']:.2f}  iters={info['iters']}  "
                              f"conv={info['converged']}  ({dt:.1f}s)", flush=True)
                    else:
                        print(f"    K={K} {status} ({dt:.1f}s)", flush=True)
        else:
            for task in tasks:
                K = task[2]
                print(f"  TRAIN {cache_path(name, K)}", flush=True)
                K_, status, info, dt = _train_one_K(task)
                if status == 'OK':
                    print(f"    LL_tot={info['ll']:.2f}  iters={info['iters']}  "
                          f"converged={info['converged']}  ({dt:.1f}s)", flush=True)
                gc.collect()

        del obs
        gc.collect()

    print(f"\n[plan_b-caches] Completado. total {time.time()-total_start:.1f}s", flush=True)


if __name__ == '__main__':
    main()
