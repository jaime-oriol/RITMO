"""Fase 4 del Paper-1: Cross-domain transfer con 3 seeds.

HMM entrenado en 4 datasets fuente (ETTh1/ETTh2/Weather/Electricity) se transfiere
CONGELADO a 2 datasets objetivo (Traffic, Exchange). El transformer sí se entrena
en el objetivo. Se usa el (variant, K) óptimo robusto de cada fuente según
scripts/paper1/k_optimal.json, además de las 5 baselines deterministas como
referencia sobre los mismos datasets objetivo.

Pred horizons reducidos {96, 336} (igual que notebooks/zero_shot.ipynb).

No sobrescribe resultados previos: usa --des crossdom_paper1_{kind}_seed{s}.
Resumible. Libera memoria entre runs.

Uso:
    python -u scripts/paper1/phase4_cross_domain_seeds.py
    python -u scripts/paper1/phase4_cross_domain_seeds.py --only-target Traffic
    python -u scripts/paper1/phase4_cross_domain_seeds.py --only-seed 2021
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from scripts.paper1.config import (  # noqa: E402
    CROSS_DOMAIN_TARGETS, CROSS_DOMAIN_HORIZONS, SEEDS,
    BASELINE_TECHNIQUES, TRANSFORMER_CFG, hmm_cache_path,
)


def load_k_optimal():
    path = Path('scripts/paper1/k_optimal.json')
    if not path.exists():
        raise FileNotFoundError(
            f"{path} no existe. Ejecuta scripts/paper1/phase2_k_sweep_seeds.py primero."
        )
    return json.loads(path.read_text())


def exp_des_baseline(tgt_name: str, technique: str, seed: int) -> str:
    return f"crossdom_paper1_{tgt_name}_{technique}_seed{seed}"


def exp_des_hmm(tgt_name: str, src_name: str, variant: str, K: int, seed: int) -> str:
    return f"crossdom_paper1_{tgt_name}_from_{src_name}_{variant}_K{K}_seed{seed}"


def setting_str(tgt_cfg, des, pred_len):
    cfg = TRANSFORMER_CFG
    # Cross-domain usa train_epochs reducido del target (10 vs 30).
    # El setting no cambia; solo el des.
    return (
        f"plan_a_{tgt_cfg['data_arg']}_96_{pred_len}"
        f"_TransformerCommon_{tgt_cfg['data_arg']}"
        f"_ftS_sl{cfg['seq_len']}_ll{cfg['label_len']}_pl{pred_len}"
        f"_dm{cfg['d_model']}_nh{cfg['n_heads']}_el{cfg['e_layers']}_dl1"
        f"_df{cfg['d_ff']}_expand2_dc4_fc1_ebtimeF_dtTrue_{des}_0"
    )


def result_metrics_path(tgt_cfg, des, pred_len):
    return Path('results') / setting_str(tgt_cfg, des, pred_len) / 'metrics.npy'


def build_cmd(tgt_cfg, des, pred_len, seed, technique, K, hmm_cache):
    cfg = TRANSFORMER_CFG
    cmd = [
        sys.executable, '-u', 'run.py',
        '--task_name', 'plan_a', '--is_training', '1',
        '--root_path', tgt_cfg['root'], '--data_path', tgt_cfg['data_path'],
        '--model_id', f"{tgt_cfg['data_arg']}_96_{pred_len}",
        '--model', 'TransformerCommon', '--data', tgt_cfg['data_arg'],
        '--features', 'S', '--target', tgt_cfg['target'], '--freq', tgt_cfg['freq'],
        '--seq_len', str(cfg['seq_len']), '--label_len', str(cfg['label_len']),
        '--pred_len', str(pred_len),
        '--enc_in', '1', '--dec_in', '1', '--c_out', '1',
        '--d_model', str(cfg['d_model']), '--n_heads', str(cfg['n_heads']),
        '--e_layers', str(cfg['e_layers']), '--d_ff', str(cfg['d_ff']),
        '--dropout', str(cfg['dropout']),
        '--batch_size', str(tgt_cfg['batch_size']),
        '--learning_rate', str(cfg['learning_rate']), '--lradj', cfg['lradj'],
        '--train_epochs', str(tgt_cfg['train_epochs']),
        '--patience', str(tgt_cfg['patience']),
        '--use_gpu', '0', '--num_workers', '0',
        '--technique', technique,
        '--des', des, '--itr', '1',
        '--seed', str(seed),
    ]
    if K is not None:
        cmd += ['--hmm_k', str(K), '--hmm_cache_path', hmm_cache]
    return cmd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only-target', default=None,
                    help='Traffic | Exchange')
    ap.add_argument('--only-seed', type=int, default=None)
    ap.add_argument('--only-horizons', nargs='*', type=int, default=None)
    ap.add_argument('--only-source', default=None,
                    help='ETTh1 | ETTh2 | Weather | Electricity (fuente HMM)')
    ap.add_argument('--skip-baselines', action='store_true')
    return ap.parse_args()


def run_one(cmd, env):
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        tail = '\n'.join(proc.stderr.strip().splitlines()[-10:])
        return {'status': 'fail', 'time': dt, 'err': tail}
    mse_line = [l for l in proc.stdout.splitlines() if l.startswith('mse:')]
    return {'status': 'ok', 'time': dt, 'mse_line': mse_line[-1] if mse_line else 'ok'}


def main():
    args = parse_args()
    k_optimal = load_k_optimal()

    targets = CROSS_DOMAIN_TARGETS if not args.only_target else \
              [t for t in CROSS_DOMAIN_TARGETS if t['name'] == args.only_target]
    seeds = [args.only_seed] if args.only_seed else SEEDS
    horizons = args.only_horizons if args.only_horizons else CROSS_DOMAIN_HORIZONS
    sources = list(k_optimal.keys()) if not args.only_source else [args.only_source]

    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('MKL_NUM_THREADS', '4')
    env.setdefault('OPENBLAS_NUM_THREADS', '4')

    # Construir lista completa de experimentos.
    experiments = []
    for tgt in targets:
        # Baselines (deterministas pero varían con seed por init del Transformer).
        if not args.skip_baselines:
            for tech in BASELINE_TECHNIQUES:
                for pred_len in horizons:
                    for seed in seeds:
                        des = exp_des_baseline(tgt['name'], tech, seed)
                        experiments.append({
                            'tag': f"{tgt['name']} {tech} pl={pred_len} seed={seed}",
                            'tgt': tgt, 'des': des, 'pred_len': pred_len, 'seed': seed,
                            'technique': tech, 'K': None, 'hmm_cache': None,
                        })
        # HMM cross-domain: por fuente con K óptimo robusto.
        for src_name in sources:
            opt = k_optimal.get(src_name, {})
            if not opt.get('variant'):
                continue
            variant = opt['variant']
            K = int(opt['K'])
            for pred_len in horizons:
                for seed in seeds:
                    des = exp_des_hmm(tgt['name'], src_name, variant, K, seed)
                    experiments.append({
                        'tag': f"{tgt['name']} HMM<-{src_name} {variant} K={K} pl={pred_len} seed={seed}",
                        'tgt': tgt, 'des': des, 'pred_len': pred_len, 'seed': seed,
                        'technique': variant, 'K': K,
                        'hmm_cache': hmm_cache_path(src_name, K, seed),
                    })

    total = len(experiments)
    done = ok = skipped = failed = 0
    t_all = time.time()
    print(f"[paper1-phase4] targets={[t['name'] for t in targets]} "
          f"sources={sources} seeds={seeds} horizons={horizons} total={total}", flush=True)

    for exp in experiments:
        done += 1
        # Skip si existe.
        if result_metrics_path(exp['tgt'], exp['des'], exp['pred_len']).exists():
            skipped += 1
            print(f"[{done}/{total}] SKIP {exp['tag']}", flush=True)
            continue
        # Verificar cache HMM si aplica.
        if exp['hmm_cache'] and not Path(exp['hmm_cache']).exists():
            print(f"[{done}/{total}] SKIP (no HMM cache) {exp['tag']}", flush=True)
            failed += 1
            continue

        print(f"[{done}/{total}] RUN  {exp['tag']} ...", end=' ', flush=True)
        cmd = build_cmd(
            exp['tgt'], exp['des'], exp['pred_len'], exp['seed'],
            exp['technique'], exp['K'], exp['hmm_cache'],
        )
        r = run_one(cmd, env)
        if r['status'] == 'ok':
            ok += 1
            print(f"({r['time']:.0f}s) {r['mse_line']}", flush=True)
        else:
            failed += 1
            print(f"FAIL ({r['time']:.0f}s)\n{r['err']}", flush=True)
        gc.collect()

    print(f"\n[paper1-phase4] Fin. ok={ok} skipped={skipped} failed={failed} "
          f"total={done}  wall={time.time()-t_all:.1f}s", flush=True)


if __name__ == '__main__':
    main()
