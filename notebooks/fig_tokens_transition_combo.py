"""Genera la figura de tokenizacion estilo fig20 (scatter de regimenes + secuencia
+ distribucion) PERO anadiendo la matriz de transicion A. Para Weather K=4 y ETTh2 K=5."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

REPO = Path('/home/jaime/TFG/RITMO')
os.chdir(REPO); sys.path.insert(0, str(REPO))
from hmm import forward_backward_batch

# --- estilo identico a paper1_figures.py ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3
sns.set_style('whitegrid')

OUT = REPO / 'memoria/secciones/figures_paper1'


def build(cache_path, csv_path, title_name, out_name, n_windows=20, seq_len=96):
    cache = torch.load(cache_path, weights_only=False)
    g = lambda x: x.numpy() if isinstance(x, torch.Tensor) else x
    A, pi, mu, sigma = g(cache['A']), g(cache['pi']), g(cache['mu']), g(cache['sigma'])
    K = len(mu)

    values = pd.read_csv(csv_path)['OT'].values.astype(float)
    test_start = int(len(values) * 0.8)
    test_raw = values[test_start:]
    wins = []
    for s in range(0, len(test_raw) - seq_len + 1, seq_len):
        w = test_raw[s:s + seq_len]
        wins.append((w - w.mean()) / max(w.std(), 1e-5))
    segment = np.concatenate(wins[:n_windows])

    gamma, _, _ = forward_backward_batch(segment.reshape(1, -1), A, pi, mu, sigma, need_xi=False)
    states = np.argmax(gamma[0], axis=1)

    order = np.argsort(mu)
    smap = {old: new for new, old in enumerate(order)}
    states = np.array([smap[s] for s in states])
    mu_s, sigma_s = mu[order], sigma[order]
    A_s = A[order][:, order]
    colors = [plt.cm.tab10(i) for i in range(K)]
    T = len(segment)

    # --- layout: serie arriba (ancho completo), abajo [secuencia | matriz A | distribucion] ---
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.3, 1.5], width_ratios=[2.1, 1, 1],
                          hspace=0.32, wspace=0.28)
    ax_series = fig.add_subplot(gs[0, :])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_mat = fig.add_subplot(gs[1, 1])
    ax_stats = fig.add_subplot(gs[1, 2])

    # (a) serie con regimenes
    ax_series.plot(segment, color='gray', linewidth=0.6, alpha=0.4, zorder=1)
    for k in range(K):
        mask = states == k
        if np.any(mask):
            ax_series.scatter(np.where(mask)[0], segment[mask], color=colors[k], s=8,
                              alpha=0.7, zorder=2,
                              label=f'S{k} ($\\mu$={mu_s[k]:.2f}, $\\sigma$={sigma_s[k]:.2f})')
            ax_series.axhline(mu_s[k], color=colors[k], linestyle='--', linewidth=1, alpha=0.4)
    changes = np.sum(states[1:] != states[:-1])
    cr = T / (changes + 1)
    ax_series.set_ylabel('Valor normalizado')
    ax_series.set_title(f'{title_name} — Regimenes HMM (K = {K}) | {changes+1} segmentos (CR = {cr:.1f}x)',
                        fontweight='bold', fontsize=12)
    ax_series.legend(loc='upper right', fontsize=7, ncol=K, framealpha=0.9)
    ax_series.grid(alpha=0.3)

    # (b) secuencia de tokens
    prev, start = states[0], 0
    for t in range(1, T):
        if states[t] != prev or t == T - 1:
            ax_bar.axvspan(start, t, color=colors[prev], alpha=0.9)
            start, prev = t, states[t]
    ax_bar.set_xlim(0, T); ax_bar.set_yticks([])
    ax_bar.set_xlabel('Timestep')
    ax_bar.set_title('Secuencia de tokens', fontweight='bold', fontsize=10)

    # (c) matriz de transicion A
    im = ax_mat.imshow(A_s, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    for i in range(K):
        for j in range(K):
            v = A_s[i, j]
            ax_mat.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if v > 0.5 else 'black', fontweight='bold')
    ax_mat.set_xticks(range(K)); ax_mat.set_yticks(range(K))
    ax_mat.set_xticklabels([f'S{i}' for i in range(K)])
    ax_mat.set_yticklabels([f'S{i}' for i in range(K)])
    ax_mat.set_xlabel('A estado'); ax_mat.set_ylabel('Desde estado')
    diag = np.mean(np.diag(A_s))
    ax_mat.set_title(f'Matriz de transicion A (diag. media = {diag:.2f})',
                     fontweight='bold', fontsize=10)
    fig.colorbar(im, ax=ax_mat, shrink=0.8, label='P(i->j)')

    # (d) distribucion
    counts = [np.sum(states == k) for k in range(K)]
    ax_stats.barh(range(K), counts, color=colors, alpha=0.8, edgecolor='black')
    ax_stats.set_yticks(range(K)); ax_stats.set_yticklabels([f'S{k}' for k in range(K)])
    ax_stats.set_xlabel('Frecuencia')
    ax_stats.set_title('Distribucion', fontweight='bold', fontsize=10)
    ax_stats.grid(alpha=0.3, axis='x')
    for k in range(K):
        ax_stats.text(counts[k], k, f' {100*counts[k]/T:.1f}%', va='center', fontsize=8)

    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'{out_name}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    print(f'saved: {out_name}.{{png,svg}}  (K={K}, CR={cr:.1f}x, diag={diag:.2f})')
    plt.close(fig)


build(REPO / 'cache/hmm_weather_K4_seed42.pth', REPO / 'dataset/weather/weather.csv',
      'Weather', 'fig20b_weather_K4_tokens_transition')
build(REPO / 'cache/hmm_etth2_K5_seed42.pth', REPO / 'dataset/ETT-small/ETTh2.csv',
      'ETTh2', 'fig08b_etth2_K5_tokens_transition')
print('DONE')
