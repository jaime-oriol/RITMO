"""Fase 3 del Paper-1: Plan A completo con 3 seeds.

Corre las 6 técnicas × 4 datasets × 4 horizontes × 3 seeds:
    - 5 baselines determinísticos (SAX, LLMTime, Patching, Decomposition, Foundation)
    - 1 HMM con variante + K óptimo robusto leído de scripts/paper1/k_optimal.json

Prerequisito: haber ejecutado phase1 (cachés HMM con 3 seeds) y phase2 (k_optimal.json).
Si el K óptimo coincide con el del Plan A original (seed 42), la seed 42 se reutiliza
desde results/ existente (no se re-ejecuta).

No sobrescribe resultados previos: usa --des final_paper1_{technique}_seed{s} (con K si HMM).
Resumible. Libera memoria entre runs.

Uso:
    python -u scripts/paper1/phase3_plan_a_seeds.py
    python -u scripts/paper1/phase3_plan_a_seeds.py --only-dataset ETTh1
    python -u scripts/paper1/phase3_plan_a_seeds.py --only-seed 2021
    python -u scripts/paper1/phase3_plan_a_seeds.py --only-horizons 96 192
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
    DATASETS, SEEDS, HORIZONS, BASELINE_TECHNIQUES, TRANSFORMER_CFG,
    hmm_cache_path,
)


def load_k_optimal():
    path = Path('scripts/paper1/k_optimal.json')
    if not path.exists():
        raise FileNotFoundError(
            f"{path} no existe. Ejecuta scripts/paper1/phase2_k_sweep_seeds.py primero."
        )
    return json.loads(path.read_text())


def exp_des(technique: str, K, seed: int) -> str:
    if K is not None:
        return f"final_paper1_{technique}_K{K}_seed{seed}"
    return f"final_paper1_{technique}_seed{seed}"


def setting_str(ds, technique, K, pred_len, seed):
    des = exp_des(technique, K, seed)
    cfg = TRANSFORMER_CFG
    return (
        f"plan_a_{ds['data_arg']}_96_{pred_len}"
        f"_TransformerCommon_{ds['data_arg']}"
        f"_ftS_sl{cfg['seq_len']}_ll{cfg['label_len']}_pl{pred_len}"
        f"_dm{cfg['d_model']}_nh{cfg['n_heads']}_el{cfg['e_layers']}_dl1"
        f"_df{cfg['d_ff']}_expand2_dc4_fc1_ebtimeF_dtTrue_{des}_0"
    )


def result_metrics_path(ds, technique, K, pred_len, seed):
    return Path('results') / setting_str(ds, technique, K, pred_len, seed) / 'metrics.npy'


def build_cmd(ds, technique, K, pred_len, seed):
    cfg = TRANSFORMER_CFG
    cmd = [
        sys.executable, '-u', 'run.py',
        '--task_name', 'plan_a', '--is_training', '1',
        '--root_path', ds['root'], '--data_path', ds['data_path'],
        '--model_id', f"{ds['data_arg']}_96_{pred_len}",
        '--model', 'TransformerCommon', '--data', ds['data_arg'],
        '--features', 'S', '--target', ds['target'], '--freq', ds['freq'],
        '--seq_len', str(cfg['seq_len']), '--label_len', str(cfg['label_len']),
        '--pred_len', str(pred_len),
        '--enc_in', '1', '--dec_in', '1', '--c_out', '1',
        '--d_model', str(cfg['d_model']), '--n_heads', str(cfg['n_heads']),
        '--e_layers', str(cfg['e_layers']), '--d_ff', str(cfg['d_ff']),
        '--dropout', str(cfg['dropout']),
        '--batch_size', str(cfg['batch_size']),
        '--learning_rate', str(cfg['learning_rate']), '--lradj', cfg['lradj'],
        '--train_epochs', str(cfg['train_epochs']), '--patience', str(cfg['patience']),
        '--use_gpu', '0', '--num_workers', '0',
        '--technique', technique,
        '--des', exp_des(technique, K, seed), '--itr', '1',
        '--seed', str(seed),
    ]
    if K is not None:
        cmd += ['--hmm_k', str(K), '--hmm_cache_path', hmm_cache_path(ds['name'], K, seed)]
    return cmd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only-dataset', default=None)
    ap.add_argument('--only-seed', type=int, default=None)
    ap.add_argument('--only-horizons', nargs='*', type=int, default=None)
    ap.add_argument('--only-technique', default=None,
                    help='hmm | discretization | text_based | patching | decomposition | foundation')
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

    datasets = DATASETS if not args.only_dataset else [d for d in DATASETS if d['name'] == args.only_dataset]
    seeds = [args.only_seed] if args.only_seed else SEEDS
    horizons = args.only_horizons if args.only_horizons else HORIZONS

    # Lista de (technique, K) a correr por dataset: 5 baselines + 1 HMM óptimo.
    def techniques_for(ds_name):
        techs = [(b, None) for b in BASELINE_TECHNIQUES]
        opt = k_optimal.get(ds_name, {})
        if opt.get('variant') and opt.get('K'):
            techs.append((opt['variant'], int(opt['K'])))
        else:
            print(f"[paper1-phase3] AVISO: {ds_name} sin K óptimo en k_optimal.json, se salta HMM.",
                  flush=True)
        return techs

    if args.only_technique:
        # Filtrado especial.
        def techniques_for(ds_name, _tech=args.only_technique):
            if _tech in BASELINE_TECHNIQUES:
                return [(_tech, None)]
            if _tech == 'hmm':
                opt = k_optimal.get(ds_name, {})
                if opt.get('variant'):
                    return [(opt['variant'], int(opt['K']))]
                return []
            return []

    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('MKL_NUM_THREADS', '4')
    env.setdefault('OPENBLAS_NUM_THREADS', '4')

    # Contar total.
    total = 0
    for ds in datasets:
        total += len(techniques_for(ds['name'])) * len(horizons) * len(seeds)

    done = ok = skipped = failed = 0
    t_all = time.time()
    print(f"[paper1-phase3] datasets={[d['name'] for d in datasets]} "
          f"seeds={seeds} horizons={horizons} total={total}", flush=True)

    for ds in datasets:
        print(f"\n[paper1-phase3] === {ds['name']} ===", flush=True)
        techs = techniques_for(ds['name'])
        for technique, K in techs:
            for pred_len in horizons:
                for seed in seeds:
                    done += 1
                    tag = f"{ds['name']} {technique}" + (f" K={K}" if K else "") + f" pl={pred_len} seed={seed}"

                    if result_metrics_path(ds, technique, K, pred_len, seed).exists():
                        skipped += 1
                        print(f"[{done}/{total}] SKIP {tag}", flush=True)
                        continue

                    # Verificar cache HMM si aplica.
                    if K is not None and not Path(hmm_cache_path(ds['name'], K, seed)).exists():
                        print(f"[{done}/{total}] SKIP (no HMM cache) {tag}", flush=True)
                        failed += 1
                        continue

                    print(f"[{done}/{total}] RUN  {tag} ...", end=' ', flush=True)
                    cmd = build_cmd(ds, technique, K, pred_len, seed)
                    r = run_one(cmd, env)
                    if r['status'] == 'ok':
                        ok += 1
                        print(f"({r['time']:.0f}s) {r['mse_line']}", flush=True)
                    else:
                        failed += 1
                        print(f"FAIL ({r['time']:.0f}s)\n{r['err']}", flush=True)
                    gc.collect()

    print(f"\n[paper1-phase3] Fin. ok={ok} skipped={skipped} failed={failed} "
          f"total={done}  wall={time.time()-t_all:.1f}s", flush=True)


if __name__ == '__main__':
    main()
