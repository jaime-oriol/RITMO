"""Fase 2 del Paper-1: K-sweep downstream con 3 seeds para identificar K óptimo robusto.

Corre run.py --task_name plan_a sobre 4 datasets × 8 Ks × 2 variantes HMM × 3 seeds
con pred_len=96 (horizonte representativo para selección de K).

El K óptimo robusto por dataset se determina como el K que minimiza el AVG MSE
sobre las 3 seeds y las 2 variantes HMM (se elige también la variante ganadora).

Output:
    scripts/paper1/k_optimal.json  - dict con {dataset: {variant, K}} robusto

No sobrescribe resultados previos: usa --des ksweep_paper1_{variant}_K{k}_seed{s}
Resumible: salta experimentos cuyo metrics.npy ya existe.

Uso:
    python -u scripts/paper1/phase2_k_sweep_seeds.py
    python -u scripts/paper1/phase2_k_sweep_seeds.py --only-dataset ETTh1
    python -u scripts/paper1/phase2_k_sweep_seeds.py --only-seed 2021
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from scripts.paper1.config import (  # noqa: E402
    DATASETS, K_VALUES, SEEDS, HMM_VARIANTS, TRANSFORMER_CFG, hmm_cache_path,
)

# Horizonte único para K-sweep (representativo, igual que el notebook k_sweep.ipynb).
KSWEEP_PRED_LEN = 96


def exp_des(variant: str, K: int, seed: int) -> str:
    """Tag único por run, no colisiona con Plan A original."""
    return f"ksweep_paper1_{variant}_K{K}_seed{seed}"


def setting_str(ds, variant, K, seed):
    des = exp_des(variant, K, seed)
    cfg = TRANSFORMER_CFG
    # Formato exacto del setting de run.py.
    return (
        f"plan_a_{ds['data_arg']}_96_{KSWEEP_PRED_LEN}"
        f"_TransformerCommon_{ds['data_arg']}"
        f"_ftS_sl{cfg['seq_len']}_ll{cfg['label_len']}_pl{KSWEEP_PRED_LEN}"
        f"_dm{cfg['d_model']}_nh{cfg['n_heads']}_el{cfg['e_layers']}_dl1"
        f"_df{cfg['d_ff']}_expand2_dc4_fc1_ebtimeF_dtTrue_{des}_0"
    )


def result_metrics_path(ds, variant, K, seed):
    return Path('results') / setting_str(ds, variant, K, seed) / 'metrics.npy'


def build_cmd(ds, variant, K, seed):
    cp = hmm_cache_path(ds['name'], K, seed)
    cfg = TRANSFORMER_CFG
    return [
        sys.executable, '-u', 'run.py',
        '--task_name', 'plan_a', '--is_training', '1',
        '--root_path', ds['root'], '--data_path', ds['data_path'],
        '--model_id', f"{ds['data_arg']}_96_{KSWEEP_PRED_LEN}",
        '--model', 'TransformerCommon', '--data', ds['data_arg'],
        '--features', 'S', '--target', ds['target'], '--freq', ds['freq'],
        '--seq_len', str(cfg['seq_len']), '--label_len', str(cfg['label_len']),
        '--pred_len', str(KSWEEP_PRED_LEN),
        '--enc_in', '1', '--dec_in', '1', '--c_out', '1',
        '--d_model', str(cfg['d_model']), '--n_heads', str(cfg['n_heads']),
        '--e_layers', str(cfg['e_layers']), '--d_ff', str(cfg['d_ff']),
        '--dropout', str(cfg['dropout']),
        '--batch_size', str(cfg['batch_size']),
        '--learning_rate', str(cfg['learning_rate']), '--lradj', cfg['lradj'],
        '--train_epochs', str(cfg['train_epochs']), '--patience', str(cfg['patience']),
        '--use_gpu', '0', '--num_workers', '0',
        '--technique', variant, '--hmm_k', str(K),
        '--hmm_cache_path', cp,
        '--des', exp_des(variant, K, seed), '--itr', '1',
        '--seed', str(seed),
    ]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only-dataset', default=None)
    ap.add_argument('--only-seed', type=int, default=None)
    ap.add_argument('--only-ks', nargs='*', type=int, default=None)
    ap.add_argument('--only-variants', nargs='*', default=None)
    ap.add_argument('--aggregate-only', action='store_true',
                    help='No ejecuta runs, solo agrega resultados existentes a k_optimal.json.')
    return ap.parse_args()


def run_one(ds, variant, K, seed, env):
    cmd = build_cmd(ds, variant, K, seed)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        tail = '\n'.join(proc.stderr.strip().splitlines()[-10:])
        return {'status': 'fail', 'time': dt, 'err': tail}
    # Extraer MSE/MAE del stdout.
    mse_line = [l for l in proc.stdout.splitlines() if l.startswith('mse:')]
    return {'status': 'ok', 'time': dt, 'mse_line': mse_line[-1] if mse_line else 'ok'}


def aggregate_to_k_optimal():
    """Lee todos los metrics.npy existentes y decide el K óptimo robusto por dataset.

    Regla: K óptimo = argmin_{variant, K} avg_{seed} MSE.
    """
    k_optimal = {}
    for ds in DATASETS:
        best = {'variant': None, 'K': None, 'mse_avg': np.inf, 'mse_per_seed': None}
        for variant in HMM_VARIANTS:
            for K in K_VALUES:
                mses = []
                for seed in SEEDS:
                    mp = result_metrics_path(ds, variant, K, seed)
                    if not mp.exists():
                        continue
                    m = np.load(mp)  # [mae, mse, rmse, mape, mspe]
                    mses.append(float(m[1]))
                if len(mses) < 2:  # necesitamos al menos 2 seeds para avg
                    continue
                mse_avg = float(np.mean(mses))
                if mse_avg < best['mse_avg']:
                    best = {
                        'variant': variant, 'K': K,
                        'mse_avg': mse_avg,
                        'mse_per_seed': mses,
                        'n_seeds': len(mses),
                    }
        k_optimal[ds['name']] = best

    out = Path('scripts/paper1/k_optimal.json')
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(k_optimal, indent=2))
    print(f"\n[paper1-phase2] Escrito {out}:")
    for name, best in k_optimal.items():
        if best['variant']:
            print(f"  {name}: {best['variant']} K={best['K']} "
                  f"avg_MSE={best['mse_avg']:.6f} ({best['n_seeds']} seeds)")
        else:
            print(f"  {name}: SIN DATOS suficientes")


def main():
    args = parse_args()

    if args.aggregate_only:
        aggregate_to_k_optimal()
        return

    datasets = DATASETS if not args.only_dataset else [d for d in DATASETS if d['name'] == args.only_dataset]
    seeds = [args.only_seed] if args.only_seed else SEEDS
    ks = args.only_ks if args.only_ks else K_VALUES
    variants = args.only_variants if args.only_variants else HMM_VARIANTS

    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('MKL_NUM_THREADS', '4')
    env.setdefault('OPENBLAS_NUM_THREADS', '4')

    total = len(datasets) * len(variants) * len(ks) * len(seeds)
    done = ok = skipped = failed = 0
    t_all = time.time()

    print(f"[paper1-phase2] datasets={[d['name'] for d in datasets]} "
          f"variants={variants} Ks={ks} seeds={seeds} total={total}", flush=True)

    for ds in datasets:
        print(f"\n[paper1-phase2] === {ds['name']} ===", flush=True)
        for variant in variants:
            for K in ks:
                for seed in seeds:
                    done += 1
                    tag = f"{ds['name']} {variant} K={K} seed={seed}"

                    # Verificar cache HMM existe.
                    if not Path(hmm_cache_path(ds['name'], K, seed)).exists():
                        print(f"[{done}/{total}] SKIP (no HMM cache) {tag}", flush=True)
                        failed += 1
                        continue

                    # Verificar resultado existe.
                    if result_metrics_path(ds, variant, K, seed).exists():
                        skipped += 1
                        print(f"[{done}/{total}] SKIP {tag}", flush=True)
                        continue

                    print(f"[{done}/{total}] RUN  {tag} ...", end=' ', flush=True)
                    r = run_one(ds, variant, K, seed, env)
                    if r['status'] == 'ok':
                        ok += 1
                        print(f"({r['time']:.0f}s) {r['mse_line']}", flush=True)
                    else:
                        failed += 1
                        print(f"FAIL ({r['time']:.0f}s)\n{r['err']}", flush=True)
                    gc.collect()

    print(f"\n[paper1-phase2] Runs: ok={ok} skipped={skipped} failed={failed} "
          f"wall={time.time()-t_all:.1f}s", flush=True)

    # Agregar al final.
    aggregate_to_k_optimal()


if __name__ == '__main__':
    main()
