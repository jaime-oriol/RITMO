"""Fase 5 del Paper-1: métricas intrínsecas de tokenización con 3 seeds.

Calcula las 11 métricas de tecnicas/metrics.py sobre 4 datasets × 6 técnicas × 3 seeds.
Para HMM usa el (variant, K) óptimo robusto de scripts/paper1/k_optimal.json.
Para las métricas discretas (vocabulary_entropy, bigram_entropy, token_persistence,
top_k_coverage, edit_distance_normalized) se decodifica el HMM con Viterbi.

No sobrescribe resultados previos: output en results/paper1_intrinsic/{dataset}_seed{s}.json.
Resumible. Libera memoria entre runs.

Uso:
    python -u scripts/paper1/phase5_intrinsic_metrics.py
    python -u scripts/paper1/phase5_intrinsic_metrics.py --only-dataset ETTh1
    python -u scripts/paper1/phase5_intrinsic_metrics.py --only-seed 2021
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from scripts.paper1.config import DATASETS, SEEDS, hmm_cache_path  # noqa: E402
from tecnicas.discretization import sax_discretize, decode_sax  # noqa: E402
from tecnicas.text_based import text_based_tokenize, decode_text_based  # noqa: E402
from tecnicas.patching import patching_tokenize  # noqa: E402
from tecnicas.decomposition import decomposition_tokenize  # noqa: E402
from tecnicas.foundation import (  # noqa: E402
    foundation_tokenize, reconstruct_from_patches, apply_masking,
)
from tecnicas.metrics import (  # noqa: E402
    evaluate_tokenization, perturbation_stability,
    edit_distance_normalized, token_distance_continuous,
)
from hmm.viterbi import viterbi_decode  # noqa: E402

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

SEQ_LEN = 96
OUT_DIR = Path('results/paper1_intrinsic')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_k_optimal():
    p = Path('scripts/paper1/k_optimal.json')
    if not p.exists():
        raise FileNotFoundError(
            f"{p} no existe. Ejecuta phase2_k_sweep_seeds.py primero."
        )
    return json.loads(p.read_text())


def load_train_normalized(ds):
    path = os.path.join(ds['root'], ds['data_path'])
    df = pd.read_csv(path)
    target = ds['target']
    if ds['split_type'] == 'custom':
        cols = list(df.columns)
        cols.remove(target)
        cols.remove('date')
        df = df[['date'] + cols + [target]]
    n_train = 12 * 30 * 24 if ds['split_type'] == 'ETT' else int(len(df) * 0.7)
    train_vals = df[target].values[:n_train].astype(np.float64)
    windows = []
    for start in range(0, len(train_vals) - SEQ_LEN + 1, SEQ_LEN):
        w = train_vals[start:start + SEQ_LEN]
        m, s = w.mean(), max(w.std(), 1e-5)
        windows.append((w - m) / s)
    return np.concatenate(windows)


def load_hmm_params(ds_name, K, seed):
    cp = hmm_cache_path(ds_name, K, seed)
    if not Path(cp).exists():
        raise FileNotFoundError(f"Cache HMM no encontrada: {cp}")
    cache = torch.load(cp, weights_only=False)
    return {
        'A': cache['A'].numpy() if isinstance(cache['A'], torch.Tensor) else np.asarray(cache['A']),
        'pi': cache['pi'].numpy() if isinstance(cache['pi'], torch.Tensor) else np.asarray(cache['pi']),
        'mu': cache['mu'].numpy() if isinstance(cache['mu'], torch.Tensor) else np.asarray(cache['mu']),
        'sigma': cache['sigma'].numpy() if isinstance(cache['sigma'], torch.Tensor) else np.asarray(cache['sigma']),
    }


def compute_metrics_for_dataset(ds, seed, k_optimal):
    """Calcula las 11 métricas para las 6 técnicas sobre un dataset en una seed."""
    serie_norm = load_train_normalized(ds)
    T = len(serie_norm)
    print(f"    T={T}", flush=True)

    results = {}

    # 1. Discretization (SAX)
    sax = sax_discretize(serie_norm, alphabet_size=8)
    sax_recon = decode_sax(sax['tokens'], sax['breakpoints'])
    results['discretization'] = evaluate_tokenization(
        serie_norm, sax_recon, num_tokens=sax['num_tokens'], tokens=sax['tokens'],
    )

    # 2. Text-based (LLMTime)
    llm = text_based_tokenize(serie_norm, precision=2)
    llm_recon = decode_text_based(llm['text'])
    results['text_based'] = evaluate_tokenization(
        serie_norm, llm_recon, num_tokens=len(llm['text']), tokens=None,
    )

    # 3. Patching (PatchTST)
    patches = patching_tokenize(serie_norm, patch_len=16, stride=16)
    patch_recon = patches.flatten()[:T]
    if len(patch_recon) < T:
        patch_recon = np.pad(patch_recon, (0, T - len(patch_recon)))
    results['patching'] = evaluate_tokenization(
        serie_norm, patch_recon, num_tokens=patches.shape[0], tokens=None,
    )

    # 4. Decomposition (Autoformer)
    decomp = decomposition_tokenize(serie_norm, kernel_size=25)
    decomp_recon = decomp['trend'] + decomp['seasonal']
    results['decomposition'] = evaluate_tokenization(
        serie_norm, decomp_recon, num_tokens=2, tokens=None,
    )

    # 5. Foundation (MOMENT-inspired) — seed afecta al masking aleatorio.
    moment = foundation_tokenize(
        serie_norm, patch_len=16, stride=16, mask_ratio=0.3, random_seed=seed,
    )
    masked_p = apply_masking(moment['patches'], moment['mask'], mask_value=0.0)
    moment_recon = reconstruct_from_patches(masked_p, 16, 16, T)
    results['foundation'] = evaluate_tokenization(
        serie_norm, moment_recon, num_tokens=moment['num_patches'], tokens=None,
    )

    # 6. HMM (RITMO) con K óptimo robusto de la seed actual.
    opt = k_optimal.get(ds['name'], {})
    if not opt.get('variant'):
        raise ValueError(f"No hay K óptimo para {ds['name']} en k_optimal.json")
    K = int(opt['K'])
    variant = opt['variant']
    hmm_p = load_hmm_params(ds['name'], K, seed)
    states, _ = viterbi_decode(
        serie_norm, hmm_p['A'], hmm_p['pi'], hmm_p['mu'], hmm_p['sigma'],
    )
    hmm_recon = hmm_p['mu'][states]
    results['hmm'] = evaluate_tokenization(
        serie_norm, hmm_recon, num_tokens=T, tokens=states,
    )
    results['hmm']['_variant'] = variant
    results['hmm']['_K'] = K

    return results, (serie_norm, hmm_p)


def compute_robustness(ds, seed, serie_norm, hmm_p):
    """Métricas de robustez (perturbation_stability + token distances)."""
    # Subset 20% para acelerar, igual que el notebook original.
    subset = serie_norm[:int(len(serie_norm) * 0.2)]

    def sax_pipe(s):
        r = sax_discretize(s, alphabet_size=8)
        return decode_sax(r['tokens'], r['breakpoints'])

    def text_pipe(s):
        r = text_based_tokenize(s, precision=2)
        return decode_text_based(r['text'])

    def patch_pipe(s):
        p = patching_tokenize(s, patch_len=16, stride=16)
        recon = p.flatten()[:len(s)]
        if len(recon) < len(s):
            recon = np.pad(recon, (0, len(s) - len(recon)))
        return recon

    def decomp_pipe(s):
        d = decomposition_tokenize(s, kernel_size=25)
        return d['trend'] + d['seasonal']

    def foundation_pipe(s):
        f = foundation_tokenize(s, patch_len=16, stride=16, mask_ratio=0.3, random_seed=seed)
        masked = apply_masking(f['patches'], f['mask'], mask_value=0.0)
        return reconstruct_from_patches(masked, 16, 16, len(s))

    def hmm_pipe(s):
        st, _ = viterbi_decode(s, hmm_p['A'], hmm_p['pi'], hmm_p['mu'], hmm_p['sigma'])
        return hmm_p['mu'][st]

    pipelines = {
        'discretization': sax_pipe,
        'text_based': text_pipe,
        'patching': patch_pipe,
        'decomposition': decomp_pipe,
        'foundation': foundation_pipe,
        'hmm': hmm_pipe,
    }
    robust = {}
    for name, pipe in pipelines.items():
        robust[name] = perturbation_stability(
            subset, pipe, noise_levels=[0.1, 0.5], n_reps=3, random_seed=seed,
        )

    # Token distances intra-técnica ante ruido 0.5σ.
    rng = np.random.RandomState(seed)
    noisy = subset + rng.normal(0, 0.5 * subset.std(), len(subset))

    td = {}
    # Discretas.
    sax_o = sax_discretize(subset, alphabet_size=8)['tokens']
    sax_p = sax_discretize(noisy, alphabet_size=8)['tokens']
    td['discretization_edit'] = edit_distance_normalized(sax_o, sax_p)

    h_o, _ = viterbi_decode(subset, hmm_p['A'], hmm_p['pi'], hmm_p['mu'], hmm_p['sigma'])
    h_n, _ = viterbi_decode(noisy, hmm_p['A'], hmm_p['pi'], hmm_p['mu'], hmm_p['sigma'])
    td['hmm_edit'] = edit_distance_normalized(h_o, h_n)

    # Continuas.
    p_o = patching_tokenize(subset, patch_len=16, stride=16)
    p_n = patching_tokenize(noisy, patch_len=16, stride=16)
    td['patching_l2'] = token_distance_continuous(p_o, p_n, metric='l2')

    d_o = decomposition_tokenize(subset, kernel_size=25)
    d_n = decomposition_tokenize(noisy, kernel_size=25)
    td['decomposition_l2'] = token_distance_continuous(
        np.stack([d_o['trend'], d_o['seasonal']]),
        np.stack([d_n['trend'], d_n['seasonal']]),
        metric='l2',
    )

    f_o = foundation_tokenize(subset, patch_len=16, stride=16, mask_ratio=0.3, random_seed=seed)['patches']
    f_n = foundation_tokenize(noisy, patch_len=16, stride=16, mask_ratio=0.3, random_seed=seed)['patches']
    td['foundation_l2'] = token_distance_continuous(f_o, f_n, metric='l2')

    return robust, td


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only-dataset', default=None)
    ap.add_argument('--only-seed', type=int, default=None)
    ap.add_argument('--skip-robustness', action='store_true',
                    help='Salta el bloque de perturbation_stability (más lento).')
    return ap.parse_args()


def main():
    args = parse_args()
    k_optimal = load_k_optimal()

    datasets = DATASETS if not args.only_dataset else [d for d in DATASETS if d['name'] == args.only_dataset]
    seeds = [args.only_seed] if args.only_seed else SEEDS

    total = len(datasets) * len(seeds)
    done = ok = skipped = 0
    t_all = time.time()
    print(f"[paper1-phase5] datasets={[d['name'] for d in datasets]} "
          f"seeds={seeds} total={total}", flush=True)

    for ds in datasets:
        for seed in seeds:
            done += 1
            out_path = OUT_DIR / f"{ds['name']}_seed{seed}.json"
            if out_path.exists():
                skipped += 1
                print(f"[{done}/{total}] SKIP {out_path.name}", flush=True)
                continue

            print(f"\n[{done}/{total}] {ds['name']} seed={seed}", flush=True)
            t0 = time.time()
            metrics, (serie_norm, hmm_p) = compute_metrics_for_dataset(ds, seed, k_optimal)

            if not args.skip_robustness:
                print("    perturbation_stability + token distances...", flush=True)
                robust, token_dist = compute_robustness(ds, seed, serie_norm, hmm_p)
                metrics['_robustness'] = robust
                metrics['_token_distance'] = token_dist

            out_path.write_text(json.dumps(metrics, indent=2, default=float))
            ok += 1
            print(f"    -> {out_path.name} ({time.time()-t0:.1f}s)", flush=True)
            del serie_norm, hmm_p
            gc.collect()

    print(f"\n[paper1-phase5] Fin. ok={ok} skipped={skipped} total={done} "
          f"wall={time.time()-t_all:.1f}s", flush=True)


if __name__ == '__main__':
    main()
