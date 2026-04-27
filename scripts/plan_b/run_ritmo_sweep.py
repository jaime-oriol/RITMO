"""Corre el K sweep RITMO-M del Plan B, resumible.

Cada experimento invoca run.py --task_name plan_b. Salta los ya existentes (metrics.npy).
Log completo por linea en stdout: usa `tee`/`nohup` para persistirlo.

Uso:
    python -u scripts/plan_b/run_ritmo_sweep.py
    python -u scripts/plan_b/run_ritmo_sweep.py --only-dataset ETTh1
    python -u scripts/plan_b/run_ritmo_sweep.py --skip-electricity --horizons 96 192
"""
import argparse
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from scripts.plan_b.plan_b_config import (  # noqa: E402
    DATASETS, K_VALUES_BY_DATASET, VARIANTS, HORIZONS, SEED,
    cache_path, experiment_tag,
)


def result_already_exists(tag: str) -> bool:
    """True si existe results/plan_b_{tag}_..._{tag}_0/metrics.npy."""
    for p in Path('results').glob(f'plan_b_{tag}_*_{tag}_0'):
        if (p / 'metrics.npy').exists():
            return True
    return False


def build_cmd(ds, variant, K, pred_len, use_gpu=0):
    tag = experiment_tag(ds['name'], variant, K, pred_len)
    cp = cache_path(ds['name'], K)
    return tag, cp, [
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
        '--use_gpu', str(use_gpu), '--num_workers', '0',
        '--technique', variant, '--hmm_k', str(K),
        '--hmm_cache_path', cp,
        '--des', tag, '--itr', '1',
    ]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only-dataset', default=None,
                    help='Limita a un dataset (ETTh1 | ETTh2 | Weather | Electricity).')
    ap.add_argument('--skip-electricity', action='store_true')
    ap.add_argument('--horizons', nargs='*', type=int, default=None)
    ap.add_argument('--variants', nargs='*', default=None)
    ap.add_argument('--ks', nargs='*', type=int, default=None,
                    help='Override manual de K values. Default: los de plan_b_config por dataset.')
    ap.add_argument('--use-gpu', type=int, default=0,
                    help='1 para GPU, 0 para CPU. Default 0.')
    return ap.parse_args()


def main():
    args = parse_args()

    datasets = DATASETS
    if args.only_dataset:
        datasets = [d for d in DATASETS if d['name'] == args.only_dataset]
    if args.skip_electricity:
        datasets = [d for d in datasets if d['name'] != 'Electricity']

    horizons = args.horizons if args.horizons else HORIZONS
    variants = args.variants if args.variants else VARIANTS

    # Limitar threads (WSL-safe)
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('MKL_NUM_THREADS', '4')
    env.setdefault('OPENBLAS_NUM_THREADS', '4')

    total_runs = sum(len(args.ks or K_VALUES_BY_DATASET[d['name']]) for d in datasets) \
        * len(variants) * len(horizons)
    done = skipped = failed = 0
    t_all = time.time()
    print(f"[plan_b-sweep] datasets={[d['name'] for d in datasets]} "
          f"variants={variants} horizons={horizons} total_expected={total_runs}",
          flush=True)

    for ds in datasets:
        Ks = args.ks if args.ks else K_VALUES_BY_DATASET[ds['name']]
        for K in Ks:
            cp = cache_path(ds['name'], K)
            if not Path(cp).exists():
                print(f"[plan_b-sweep] AVISO: falta cache {cp}. Correr train_hmm_caches.py antes.",
                      flush=True)
                failed += len(variants) * len(horizons)
                continue
            for variant in variants:
                for pred_len in horizons:
                    tag, _, cmd = build_cmd(ds, variant, K, pred_len, use_gpu=args.use_gpu)
                    if result_already_exists(tag):
                        skipped += 1
                        print(f"[plan_b-sweep] SKIP {tag}", flush=True)
                        continue
                    print(f"\n[plan_b-sweep] RUN {tag}  bs={ds['batch_size']} "
                          f"ep={ds['train_epochs']}", flush=True)
                    t0 = time.time()
                    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
                    dt = time.time() - t0
                    if proc.returncode != 0:
                        failed += 1
                        # Imprimir tail de stderr para diagnostico
                        err_tail = '\n'.join(proc.stderr.strip().splitlines()[-12:])
                        print(f"[plan_b-sweep] FAIL {tag} ({dt:.1f}s)\n{err_tail}",
                              flush=True)
                    else:
                        done += 1
                        metric_lines = [ln for ln in proc.stdout.splitlines()
                                        if 'mse:' in ln]
                        msg = metric_lines[-1] if metric_lines else 'OK'
                        print(f"[plan_b-sweep] OK {tag} ({dt:.1f}s)  {msg}", flush=True)
                    gc.collect()

    total = done + skipped + failed
    print(f"\n[plan_b-sweep] Fin. done={done} skipped={skipped} failed={failed} "
          f"total={total}  wall={time.time()-t_all:.1f}s", flush=True)


if __name__ == '__main__':
    main()
