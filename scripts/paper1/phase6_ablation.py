"""Fase 6 del Paper-1: ablation de componentes del embedding estructurado.

Aisla la contribución individual de cada componente de e_k = [μ_k, σ_k, A[k,·]]
al MSE downstream, manteniendo todo lo demás del pipeline constante.

Diseño:
    - Dataset:      Weather (donde HMM domina con la señal más limpia)
    - HMM:          K = 4, variante soft (no soft-residual, para aislar el efecto
                    de los componentes del embedding e_k de §3.5)
    - Horizonte:    pl = 96 (suficiente para detectar el efecto sin multiplicar coste)
    - Seeds:        {42, 2021, 7}
    - Ablations:    3 configuraciones del embedding e_k:
                      * mu_only    : e_k = [μ_k]              ∈ ℝ¹
                      * mu_sigma   : e_k = [μ_k, σ_k]         ∈ ℝ²
                      * full       : e_k = [μ_k, σ_k, A[k,·]] ∈ ℝ^{2+K}  (default)

Total: 3 ablations × 3 seeds = 9 controlled runs.

Reporta: MSE @ pl=96, mean ± std sobre 3 seeds para cada ablation.

Estado: stub — la implementación requiere las siguientes modificaciones:
    1. embeddings/embedding_generator.py — añadir param `component_mask` a
       forward_soft() que enmascara dimensiones del embedding_table.
    2. run.py — añadir flag --embedding_ablation {mu_only, mu_sigma, full}.
    3. exp/exp_plan_a.py — propagar el flag a EmbeddingGenerator.
    4. Construir cmd como en phase3_plan_a_seeds.py pero con --embedding_ablation.

Uso (cuando esté implementado):
    python -u scripts/paper1/phase6_ablation.py
    python -u scripts/paper1/phase6_ablation.py --only-ablation mu_only
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

from scripts.paper1.config import (  # noqa: E402
    DATASETS, SEEDS, TRANSFORMER_CFG, hmm_cache_path,
)

ABLATIONS = ['mu_only', 'mu_sigma', 'full']
TARGET_DATASET = 'Weather'
TARGET_K = 4
TARGET_VARIANT = 'hmm_soft'  # not soft-residual: aislamos componentes de e_k
TARGET_HORIZON = 96


def setting_str(ds_cfg, des, pred_len):
    cfg = TRANSFORMER_CFG
    return (
        f"plan_a_{ds_cfg['data_arg']}_96_{pred_len}"
        f"_TransformerCommon_{ds_cfg['data_arg']}"
        f"_ftS_sl{cfg['seq_len']}_ll{cfg['label_len']}_pl{pred_len}"
        f"_dm{cfg['d_model']}_nh{cfg['n_heads']}_el{cfg['e_layers']}_dl1"
        f"_df{cfg['d_ff']}_expand2_dc4_fc1_ebtimeF_dtTrue_{des}_0"
    )


def build_cmd(ds_cfg, des, pred_len, seed, K, hmm_cache, ablation):
    cfg = TRANSFORMER_CFG
    cmd = [
        sys.executable, '-u', 'run.py',
        '--task_name', 'plan_a', '--is_training', '1',
        '--root_path', ds_cfg['root'], '--data_path', ds_cfg['data_path'],
        '--model_id', f"{ds_cfg['data_arg']}_96_{pred_len}",
        '--model', 'TransformerCommon', '--data', ds_cfg['data_arg'],
        '--features', 'S', '--target', ds_cfg['target'], '--freq', ds_cfg['freq'],
        '--seq_len', str(cfg['seq_len']), '--label_len', str(cfg['label_len']),
        '--pred_len', str(pred_len),
        '--enc_in', '1', '--dec_in', '1', '--c_out', '1',
        '--d_model', str(cfg['d_model']), '--n_heads', str(cfg['n_heads']),
        '--e_layers', str(cfg['e_layers']), '--d_ff', str(cfg['d_ff']),
        '--dropout', str(cfg['dropout']),
        '--batch_size', str(cfg['batch_size']),
        '--learning_rate', str(cfg['learning_rate']), '--lradj', cfg['lradj'],
        '--train_epochs', str(cfg['train_epochs']),
        '--patience', str(cfg['patience']),
        '--use_gpu', '0', '--num_workers', '0',
        '--technique', TARGET_VARIANT,
        '--hmm_k', str(K),
        '--hmm_cache_path', hmm_cache,
        '--embedding_ablation', ablation,   # <-- TODO: requiere añadir este flag a run.py
        '--des', des, '--itr', '1',
        '--seed', str(seed),
    ]
    return cmd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only-ablation', default=None,
                    help='mu_only | mu_sigma | full (corre solo esta ablation)')
    ap.add_argument('--only-seed', type=int, default=None)
    ap.add_argument('--dry-run', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()

    ds_cfg = next(d for d in DATASETS if d['name'] == TARGET_DATASET)
    seeds = [args.only_seed] if args.only_seed else SEEDS
    ablations = [args.only_ablation] if args.only_ablation else ABLATIONS

    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('MKL_NUM_THREADS', '4')

    experiments = []
    for ablation in ablations:
        for seed in seeds:
            des = f"ablation_paper1_{TARGET_DATASET}_{TARGET_VARIANT}_K{TARGET_K}_{ablation}_seed{seed}"
            experiments.append({
                'tag': f"{TARGET_DATASET} {TARGET_VARIANT} K={TARGET_K} ablation={ablation} seed={seed}",
                'des': des,
                'seed': seed,
                'ablation': ablation,
                'hmm_cache': hmm_cache_path(TARGET_DATASET, TARGET_K, seed),
            })

    print(f"[paper1-phase6] dataset={TARGET_DATASET} variant={TARGET_VARIANT} "
          f"K={TARGET_K} pl={TARGET_HORIZON} ablations={ablations} seeds={seeds} "
          f"total={len(experiments)}", flush=True)

    if args.dry_run:
        for exp in experiments:
            print(f"  [DRY] {exp['tag']}")
        print("\n[paper1-phase6] dry-run completo. Salida sin ejecutar.")
        return

    # Verificar prerequisitos antes de correr
    print("\n[paper1-phase6] Prerequisites check:")
    needs_implementation = False
    # 1) flag --embedding_ablation existe en run.py?
    run_py = (REPO_ROOT / 'run.py').read_text()
    if '--embedding_ablation' not in run_py:
        print("  ✗ FALTA: añadir --embedding_ablation flag a run.py")
        needs_implementation = True
    else:
        print("  ✓ run.py acepta --embedding_ablation")
    # 2) embedding_generator soporta component_mask?
    eg = (REPO_ROOT / 'embeddings' / 'embedding_generator.py').read_text()
    if 'component_mask' not in eg and 'embedding_ablation' not in eg:
        print("  ✗ FALTA: modificar EmbeddingGenerator.forward_soft para aceptar mask")
        needs_implementation = True
    else:
        print("  ✓ EmbeddingGenerator soporta ablation")

    if needs_implementation:
        print("\n[paper1-phase6] Stub aún sin implementación. Pasos:")
        print("  1. embeddings/embedding_generator.py: añadir self.embedding_mask")
        print("     y aplicarlo en forward_soft() antes de gamma @ embedding_table.")
        print("  2. run.py: argparse --embedding_ablation choices=mu_only,mu_sigma,full default=full")
        print("  3. exp/exp_plan_a.py: pasar args.embedding_ablation a EmbeddingGenerator")
        print("  4. Volver a correr este script.\n")
        return

    done = ok = failed = 0
    t_all = time.time()
    for exp in experiments:
        done += 1
        from scripts.paper1.phase6_ablation import setting_str  # for path
        out = Path('results') / setting_str(ds_cfg, exp['des'], TARGET_HORIZON) / 'metrics.npy'
        if out.exists():
            print(f"[{done}/{len(experiments)}] SKIP {exp['tag']}")
            continue
        if not Path(exp['hmm_cache']).exists():
            print(f"[{done}/{len(experiments)}] SKIP (no HMM cache) {exp['tag']}")
            failed += 1
            continue
        cmd = build_cmd(ds_cfg, exp['des'], TARGET_HORIZON, exp['seed'],
                        TARGET_K, exp['hmm_cache'], exp['ablation'])
        print(f"[{done}/{len(experiments)}] RUN  {exp['tag']} ...", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        dt = time.time() - t0
        if proc.returncode != 0:
            failed += 1
            tail = '\n'.join(proc.stderr.strip().splitlines()[-10:])
            print(f"  FAIL ({dt:.0f}s)\n{tail}")
        else:
            ok += 1
            mse_lines = [l for l in proc.stdout.splitlines() if l.startswith('mse:')]
            print(f"  OK   ({dt:.0f}s) {mse_lines[-1] if mse_lines else ''}")
        gc.collect()

    print(f"\n[paper1-phase6] Fin. ok={ok} failed={failed} total={done}  "
          f"wall={time.time()-t_all:.1f}s")


if __name__ == '__main__':
    main()
