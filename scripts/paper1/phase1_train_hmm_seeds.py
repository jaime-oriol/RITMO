"""Fase 1 del Paper-1: entrenar HMM univariante K={3..10} x 3 seeds por dataset.

Naming canónico de salida: cache/hmm_{dataset_lower}_K{k}_seed{s}.pth

Resumible: salta cache ya existentes. Log por línea en stdout.

Uso:
    python -u scripts/paper1/phase1_train_hmm_seeds.py
    python -u scripts/paper1/phase1_train_hmm_seeds.py --only-dataset ETTh1
    python -u scripts/paper1/phase1_train_hmm_seeds.py --only-seed 2021
    python -u scripts/paper1/phase1_train_hmm_seeds.py --only-ks 3 4 5
"""
import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from scripts.paper1.config import (  # noqa: E402
    DATASETS, K_VALUES, SEEDS, hmm_cache_path,
)
from hmm.baum_welch import baum_welch  # noqa: E402
from hmm.checkpoint import save_hmm_params  # noqa: E402

# Limitar threads (WSL-safe).
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

# Hiperparámetros Baum-Welch (idénticos al notebook k_sweep.ipynb del TFG).
BW_MAX_ITER = 2000
BW_EPSILON = 1e-4
SEQ_LEN = 96  # RevIN per-window del Plan A.


def load_train_normalized(ds_cfg):
    """Carga train split + RevIN per-window (seq_len=96) del target univariante.

    Reproduce exactamente la función load_train_data de notebooks/k_sweep.ipynb.
    """
    path = os.path.join(ds_cfg['root'], ds_cfg['data_path'])
    df = pd.read_csv(path)
    target = ds_cfg['target']

    # Dataset_Custom reordena columnas (target al final). ETT no reordena.
    if ds_cfg['split_type'] == 'custom':
        cols = list(df.columns)
        cols.remove(target)
        cols.remove('date')
        df = df[['date'] + cols + [target]]

    # Split train: ETT 12 meses horarios, custom 70%.
    if ds_cfg['split_type'] == 'ETT':
        n_train = 12 * 30 * 24
    else:
        n_train = int(len(df) * 0.7)

    train_vals = df[target].values[:n_train].astype(np.float64)

    # RevIN per-window: cada ventana seq_len se normaliza con su propia media/std.
    windows = []
    for start in range(0, len(train_vals) - SEQ_LEN + 1, SEQ_LEN):
        w = train_vals[start:start + SEQ_LEN]
        m, s = w.mean(), max(w.std(), 1e-5)
        windows.append((w - m) / s)
    return np.concatenate(windows)


def train_one(dataset_name: str, obs: np.ndarray, K: int, seed: int, dry: bool = False) -> dict:
    """Entrena un único HMM y lo guarda en la ruta canónica."""
    cp = hmm_cache_path(dataset_name, K, seed)
    if Path(cp).exists():
        return {'status': 'skip', 'path': cp}
    if dry:
        return {'status': 'dry', 'path': cp}

    t0 = time.time()
    result = baum_welch(
        observations=obs,
        K=K,
        max_iter=BW_MAX_ITER,
        epsilon=BW_EPSILON,
        random_state=seed,
        verbose=False,
    )
    save_hmm_params(result, cp)
    return {
        'status': 'trained',
        'path': cp,
        'll': float(result['log_likelihood']),
        'iters': int(result['n_iter']),
        'converged': bool(result['converged']),
        'time': time.time() - t0,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only-dataset', default=None,
                    help='Limita a un dataset (ETTh1/ETTh2/Weather/Electricity).')
    ap.add_argument('--only-seed', type=int, default=None,
                    help='Limita a una seed específica (42, 2021 o 7).')
    ap.add_argument('--only-ks', nargs='*', type=int, default=None,
                    help='Lista de Ks a entrenar. Default: 3..10.')
    ap.add_argument('--dry-run', action='store_true',
                    help='No entrena, solo muestra qué haría.')
    return ap.parse_args()


def main():
    args = parse_args()
    Path('cache').mkdir(exist_ok=True)

    datasets = DATASETS
    if args.only_dataset:
        datasets = [d for d in DATASETS if d['name'] == args.only_dataset]
    seeds = [args.only_seed] if args.only_seed else SEEDS
    ks = args.only_ks if args.only_ks else K_VALUES

    total = len(datasets) * len(ks) * len(seeds)
    done = skipped = trained = copied = 0
    t_all = time.time()

    print(f"[paper1-phase1] datasets={[d['name'] for d in datasets]} "
          f"Ks={ks} seeds={seeds} total_expected={total}", flush=True)

    for ds in datasets:
        name = ds['name']
        print(f"\n[paper1-phase1] === {name} ===", flush=True)

        # Cargar datos una vez por dataset (lazy: solo si hay algo que entrenar).
        obs = None

        for K in ks:
            for seed in seeds:
                cp = hmm_cache_path(name, K, seed)
                done += 1

                if Path(cp).exists():
                    skipped += 1
                    print(f"[{done}/{total}] SKIP {cp}", flush=True)
                    continue

                # Entrenar: necesita datos cargados.
                if obs is None:
                    print(f"  Loading {name} train data...", flush=True)
                    t_load = time.time()
                    obs = load_train_normalized(ds)
                    print(f"  T={len(obs)} ({time.time()-t_load:.1f}s)", flush=True)

                print(f"[{done}/{total}] TRAIN {cp} (K={K}, seed={seed})...",
                      end=' ', flush=True)
                r = train_one(name, obs, K, seed, dry=args.dry_run)
                if r['status'] == 'dry':
                    print('DRY')
                elif r['status'] == 'trained':
                    trained += 1
                    print(f"LL={r['ll']:.2f} iter={r['iters']} "
                          f"conv={r['converged']} ({r['time']:.1f}s)", flush=True)

        del obs
        gc.collect()

    print(f"\n[paper1-phase1] Fin. trained={trained} skipped={skipped} "
          f"total={done}  wall={time.time()-t_all:.1f}s", flush=True)


if __name__ == '__main__':
    main()
