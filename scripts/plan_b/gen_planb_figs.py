"""Genera fig23 (MSE × horizonte, 4 paneles, 5 modelos) y fig24 (ranking heatmap 5×4)
para Plan B, mismo estilo que fig18/fig19 de Plan A. Solo versión _es."""
import os, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

REPO = Path('/home/jaime/TFG/RITMO')
RESULTS = REPO / '_runpod_results'
OUT_DIR = REPO / 'memoria/secciones/figures_paper1'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Estilo idéntico a paper1_figures.py
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
okabe_ito = ['#000000', '#E69F00', '#56B4E9', '#009E73',
             '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

def savefig_all(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT_DIR / f'{name}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    print(f'  saved: {name}.{{png,pdf,svg}}')

# ----- Configuración -----
PRED_LENS = [96, 192, 336, 720]
DS_ORDER = ['ETTh1', 'ETTh2', 'Weather', 'Electricity']
DS_LABEL = {'ETTh1': 'ETTh1', 'ETTh2': 'ETTh2', 'Weather': 'Weather', 'Electricity': 'Electricity'}

# K-óptimo Plan B (Tabla 26 de paper1_top_tables_planB.md)
RITMO_M = {
    'ETTh1':       ('hmm_soft',          5),
    'ETTh2':       ('hmm_soft',          3),
    'Weather':     ('hmm_soft_residual', 5),
    'Electricity': ('hmm_soft_residual', 4),
}

BASELINES = ['DLinear', 'PatchTST', 'TimeMixer', 'TimeXer']
MODELS = ['RITMO-M', 'DLinear', 'PatchTST', 'TimeMixer', 'TimeXer']

# Paleta — RITMO-M color distintivo, baselines colores adyacentes
COLORS = {
    'RITMO-M':   okabe_ito[5],  # azul fuerte
    'DLinear':   okabe_ito[1],  # naranja
    'PatchTST':  okabe_ito[2],  # cian
    'TimeMixer': okabe_ito[3],  # verde
    'TimeXer':   okabe_ito[6],  # rojo-naranja
}

# Mapping dataset → prefix en sota_baselines/
DS_BASELINE_PREFIX = {'ETTh1': 'ETTh1', 'ETTh2': 'ETTh2',
                      'Weather': 'weather', 'Electricity': 'ECL'}

# ----- Carga de datos -----
data = defaultdict(lambda: defaultdict(dict))  # data[ds][model][pl] = (MSE, MAE)

# RITMO-M desde plan_b_sweep
sweep_dir = RESULTS / 'plan_b_sweep'
for ds in DS_ORDER:
    variant, K = RITMO_M[ds]
    for pl in PRED_LENS:
        # Patrón: plan_b_PLANB_{ds}_{variant}_K{K}_pl{pl}_TransformerCommon_{ds}_ftM_...
        pattern = re.compile(
            f'^plan_b_PLANB_{ds}_{variant}_K{K}_pl{pl}_TransformerCommon_'
        )
        match = None
        for d in os.listdir(sweep_dir):
            if pattern.match(d):
                match = d
                break
        if match is None:
            print(f'  [WARN] missing RITMO-M {ds} K={K} {variant} pl={pl}')
            continue
        m = np.load(sweep_dir / match / 'metrics.npy')
        data[ds]['RITMO-M'][pl] = (float(m[1]), float(m[0]))  # MSE, MAE

# Baselines desde sota_baselines
sota_dir = RESULTS / 'sota_baselines'
for ds in DS_ORDER:
    prefix = DS_BASELINE_PREFIX[ds]
    for baseline in BASELINES:
        for pl in PRED_LENS:
            # Patrón: long_term_forecast_{prefix}_96_{pl}_{baseline}_...
            pattern = re.compile(
                f'^long_term_forecast_{prefix}_96_{pl}_{baseline}_'
            )
            match = None
            for d in os.listdir(sota_dir):
                if pattern.match(d):
                    match = d
                    break
            if match is None:
                print(f'  [WARN] missing baseline {ds} {baseline} pl={pl}')
                continue
            m = np.load(sota_dir / match / 'metrics.npy')
            data[ds][baseline][pl] = (float(m[1]), float(m[0]))

# Sanity check: 5 modelos × 4 horizontes por dataset
for ds in DS_ORDER:
    for model in MODELS:
        for pl in PRED_LENS:
            assert pl in data[ds][model], f'missing {ds} {model} pl={pl}'
print('[OK] 5 modelos × 4 horizontes × 4 datasets cargados')

# =============================================================
# FIGURE 23 — MSE por horizonte, 4 paneles (estilo fig18, ES)
# =============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
n_models = len(MODELS)

for idx, ds in enumerate(DS_ORDER):
    ax = axes[idx]
    x = np.arange(len(PRED_LENS))
    width = 0.8 / n_models
    for i, model in enumerate(MODELS):
        means = [data[ds][model][pl][0] for pl in PRED_LENS]
        ax.bar(x + i * width - 0.4 + width / 2, means, width,
               label=model, color=COLORS[model], alpha=0.85,
               edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(pl) for pl in PRED_LENS])
    ax.set_xlabel('Horizonte de predicción', fontsize=10)
    ax.set_ylabel('MSE', fontsize=10)
    ax.set_title(DS_LABEL[ds], fontweight='bold', fontsize=12)

axes[0].legend(fontsize=8, loc='upper left', framealpha=0.9)
plt.tight_layout()
savefig_all(fig, 'fig23_plan_b_mse_per_horizon_es')
plt.close(fig)

# =============================================================
# FIGURE 24 — Ranking heatmap 5×4 (estilo fig19, ES)
# =============================================================
rank_matrix = np.zeros((len(MODELS), len(DS_ORDER)))
for j, ds in enumerate(DS_ORDER):
    avgs = [(model, np.mean([data[ds][model][pl][0] for pl in PRED_LENS]))
            for model in MODELS]
    avgs_sorted = sorted(avgs, key=lambda x: x[1])
    for rank, (model, _) in enumerate(avgs_sorted, 1):
        rank_matrix[MODELS.index(model), j] = rank

fig, ax = plt.subplots(figsize=(8, 4.0))
im = ax.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto',
               vmin=1, vmax=len(MODELS))
ax.set_xticks(range(len(DS_ORDER)))
ax.set_xticklabels([DS_LABEL[ds] for ds in DS_ORDER], fontsize=11)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels(MODELS, fontsize=10)

for i in range(len(MODELS)):
    for j in range(len(DS_ORDER)):
        ax.text(j, i, f'{int(rank_matrix[i,j])}', ha='center', va='center',
                fontsize=12, fontweight='bold')

ax.set_title('Ranking por avg MSE (1 = mejor) — Plan B (seed única 2021)',
             fontweight='bold', fontsize=12)
plt.colorbar(im, ax=ax, label='Posición', shrink=0.8)
plt.tight_layout()
savefig_all(fig, 'fig24_ranking_heatmap_planb_es')
plt.close(fig)

print('\n[OK] fig23 y fig24 generadas en', OUT_DIR)
