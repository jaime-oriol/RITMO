"""Verifica convergencia de las 96 caches HMM entrenadas en fase 1.

Reporta:
    - Caches que NO convergieron (conv=False).
    - Caches con iteraciones anormalmente altas (cerca del max_iter).
    - Distribución de log-likelihoods por (dataset, K).

Opción --retrain-nonconverged: re-entrena con max_iter x2 las que no convergieron.

Uso:
    python -u scripts/paper1/verify_hmm_convergence.py                  # solo informa
    python -u scripts/paper1/verify_hmm_convergence.py --retrain-nonconverged
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from scripts.paper1.config import (  # noqa: E402
    DATASETS, K_VALUES, SEEDS, hmm_cache_path,
)
from scripts.paper1.phase1_train_hmm_seeds import load_train_normalized  # noqa: E402
from hmm.baum_welch import baum_welch  # noqa: E402
from hmm.checkpoint import save_hmm_params  # noqa: E402

BW_EPSILON = 1e-4
BW_MAX_ITER_ORIGINAL = 2000
BW_MAX_ITER_RETRY = 5000  # doblado para re-entrenos


def inspect_cache(path: Path) -> dict:
    if not path.exists():
        return {'exists': False}
    c = torch.load(path, weights_only=False)
    return {
        'exists': True,
        'converged': bool(c.get('converged', False)),
        'n_iter': int(c.get('n_iter', -1)),
        'log_likelihood': float(c.get('log_likelihood', -np.inf)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrain-nonconverged', action='store_true',
                    help='Re-entrena con max_iter=5000 las que no convergieron.')
    ap.add_argument('--only-dataset', default=None)
    args = ap.parse_args()

    datasets = DATASETS if not args.only_dataset else [d for d in DATASETS if d['name'] == args.only_dataset]

    missing, not_converged, near_limit, ok = [], [], [], []
    rows = []

    for ds in datasets:
        for K in K_VALUES:
            for seed in SEEDS:
                path = Path(hmm_cache_path(ds['name'], K, seed))
                info = inspect_cache(path)
                key = (ds['name'], K, seed, str(path))
                if not info['exists']:
                    missing.append(key)
                    continue
                info['ds'] = ds['name']; info['K'] = K; info['seed'] = seed; info['path'] = str(path)
                rows.append(info)
                if not info['converged']:
                    not_converged.append(info)
                elif info['n_iter'] >= BW_MAX_ITER_ORIGINAL - 5:
                    near_limit.append(info)
                else:
                    ok.append(info)

    total = len(datasets) * len(K_VALUES) * len(SEEDS)
    present = len(rows)
    print(f"Inspeccionadas {present}/{total} caches")
    print(f"  OK convergidas:     {len(ok)}")
    print(f"  NO convergidas:     {len(not_converged)}")
    print(f"  Cerca del max_iter: {len(near_limit)}")
    print(f"  MISSING (no entrenadas aún): {len(missing)}")

    if not_converged:
        print(f"\n=== {len(not_converged)} caches NO convergidas ===")
        for info in not_converged:
            print(f"  {info['ds']:12} K={info['K']:2} seed={info['seed']:4}  "
                  f"iter={info['n_iter']}  LL={info['log_likelihood']:.2f}  {info['path']}")

    if near_limit:
        print(f"\n=== {len(near_limit)} caches cerca del max_iter (potencialmente no convergidas) ===")
        for info in near_limit:
            print(f"  {info['ds']:12} K={info['K']:2} seed={info['seed']:4}  "
                  f"iter={info['n_iter']}  LL={info['log_likelihood']:.2f}")

    # Resumen por dataset x K (LL min/max sobre seeds).
    if rows:
        print(f"\n=== Resumen LL por (dataset, K) sobre seeds ===")
        by_dsK = {}
        for r in rows:
            by_dsK.setdefault((r['ds'], r['K']), []).append(r['log_likelihood'])
        for (ds, K), lls in sorted(by_dsK.items()):
            print(f"  {ds:12} K={K:2}  n={len(lls)}  LL min={min(lls):.2f}  max={max(lls):.2f}  "
                  f"range={max(lls)-min(lls):.2f}")

    # Re-entrenar no convergidas.
    if args.retrain_nonconverged and (not_converged or near_limit):
        targets = not_converged + near_limit
        print(f"\n=== RE-ENTRENANDO {len(targets)} caches con max_iter={BW_MAX_ITER_RETRY} ===")
        # Agrupar por dataset para cargar datos una sola vez.
        by_ds = {}
        for info in targets:
            by_ds.setdefault(info['ds'], []).append(info)
        for ds_name, items in by_ds.items():
            ds = [d for d in DATASETS if d['name'] == ds_name][0]
            print(f"\n  Loading {ds_name}...")
            obs = load_train_normalized(ds)
            for info in items:
                K, seed, path = info['K'], info['seed'], info['path']
                print(f"    RETRAIN {ds_name} K={K} seed={seed}...", end=' ', flush=True)
                t0 = time.time()
                result = baum_welch(
                    observations=obs, K=K,
                    max_iter=BW_MAX_ITER_RETRY, epsilon=BW_EPSILON,
                    random_state=seed, verbose=False,
                )
                save_hmm_params(result, path)
                print(f"LL={result['log_likelihood']:.2f} iter={result['n_iter']} "
                      f"conv={result['converged']} ({time.time()-t0:.1f}s)", flush=True)

    # Exit code: 0 si todo OK, 1 si hay no-convergidas o missing.
    if not_converged or missing:
        sys.exit(1)


if __name__ == '__main__':
    main()
