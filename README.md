# RITMO

**Regime Inference via Temporal Markov Observation**

Time-series tokenization via Hidden Markov Models, evaluated against current deterministic techniques (patching, decomposition, foundation models) under a shared Transformer architecture.

Authors: Jaime Oriol Goicoechea, Julio Emilio Sandubete Galán, Ana Lazcano.

## Research question

Can the hidden states of an HMM act as structured latent embeddings that capture statistical regimes more effectively than current deterministic tokenization techniques for univariate time-series forecasting?

## Contribution

1. **HMM as a probabilistic tokenizer**: hidden states are projected to interpretable embeddings `e_k = [mu_k, sigma_k, A[k,:]]` (regime center, volatility, transition dynamics).
2. **Six HMM embedding variants**: hard (Viterbi), soft (gamma posteriors), soft-residual (intra-regime residual), augmented, split, patched.
3. **Controlled comparison framework (Plan A)**: same Transformer, same external RevIN normalization, same splits, same optimizer — only the tokenization changes.
4. **Custom, vectorized HMM implementation**: Baum-Welch, batched forward-backward and Viterbi in NumPy with Python loops removed (verified bit-for-bit against a naive reference, diff=0.0).

## RITMO pipeline

<p align="center">
<img src="./pic/Pipeline-RITMO.png" alt="RITMO pipeline" width="700"/>
</p>

```
Series x --> RevIN --> HMM (Baum-Welch, K states) --> Viterbi / Forward-Backward
                          |
                          v
                   Structured embeddings [mu_k, sigma_k, A[k,:]]
                          |
                          v
                   Transformer encoder (fixed architecture)
                          |
                          v
                   Inverse RevIN --> Prediction y_hat
```

1. **RevIN**: reversible per-instance normalization, external and shared (Kim et al., 2022).
2. **Baum-Welch**: HMM training with K states and Gaussian emissions. K-means initialization + EM until convergence (log-likelihood, epsilon=1e-4).
3. **Viterbi / Forward-Backward**: hard assignment or soft gamma posteriors.
4. **Embeddings**: per-state e_k + variants (soft, soft-residual, augmented, split, patched) in `embeddings/embedding_generator.py`.
5. **Transformer**: encoder with a FIXED architecture shared across all techniques (`models/TransformerCommon.py`).

## Main results

Fair Plan A comparison (same config `d_model=64, n_heads=4, e_layers=2, d_ff=128`, cosine LR, 30 epochs):

**Weather, pred_len=96 — HMM beats all baselines:**

| # | Technique | MSE | MAE |
|---|-----------|-----|-----|
| 1 | **hmm_soft K=8** | **0.001200** | **0.0254** |
| 2 | hmm_soft K=5 | 0.001276 | 0.0263 |
| 3 | foundation | 0.001302 | 0.0262 |
| 4 | decomposition | 0.001309 | 0.0266 |
| 5 | patching | 0.001314 | 0.0262 |

HMM beats patching by **-8.7%** MSE. The optimum K=8 is a clear sweet spot (K=5,6,7,9,10,12 all worse).

**ETTh1, ETTh2 and other horizons (summary):**

| Dataset | Horizon | Best HMM | MSE HMM | Best baseline | MSE baseline | Gap |
|---------|---------|----------|---------|---------------|--------------|-----|
| Weather | 96  | hmm_soft K=8        | 0.00120 | patching | 0.00131 | **-8.7%** |
| ETTh2   | 96  | hmm_soft_res K=5    | 0.1475  | patching | 0.1459  | +1.1% |
| ETTh1   | 96  | hmm_soft_res K=5    | 0.0583  | patching | 0.0560  | +4.2% |
| ETTh1   | 192 | hmm_soft K=5        | 0.0762  | foundation | 0.0743 | +2.6% |
| ETTh1   | 336 | hmm_soft K=5        | 0.0882  | foundation | 0.0855 | +3.2% |

Across the 5 scenarios, HMM always beats discretization and text_based. At long horizons (pl=192, pl=336) HMM beats decomposition.

## Repository structure

> Note: `dataset/`, `md/`, `memoria/` and environment artifacts (`__pycache__/`, `.ipynb_checkpoints/`, `.venv/`, IDE configs) are excluded from the repo via `.gitignore`.

```
RITMO/
├── hmm/                           # HMM module (custom, vectorized implementation)
│   ├── __init__.py
│   ├── baum_welch.py              #   EM training with k-means init
│   ├── forward_backward.py        #   Forward-backward + batched version [B,T,K]
│   ├── viterbi.py                 #   Viterbi decoding + batched version
│   ├── gaussian_emissions.py      #   Gaussian emissions in log-space
│   ├── checkpoint.py              #   save_hmm_params / load_hmm_params
│   └── utils.py                   #   log_normalize, initialize_kmeans, LOG_EPS
│
├── embeddings/                    # Embedding generation from HMM
│   ├── __init__.py
│   ├── embedding_generator.py     #   6 variants: hard, soft, soft_residual,
│   │                              #              augmented, split, patched
│   └── technique_embeddings.py    #   Embeddings for baseline techniques
│
├── tecnicas/                      # 5 deterministic tokenization techniques
│   ├── __init__.py
│   ├── discretization.py          #   SAX (Lin et al., 2007)
│   ├── text_based.py              #   LLMTime (Gruver et al., 2023)
│   ├── patching.py                #   PatchTST (Nie et al., 2023)
│   ├── decomposition.py           #   Autoformer/DLinear (Wu et al., 2021)
│   ├── foundation.py              #   MOMENT (Goswami et al., 2024)
│   └── metrics.py                 #   Intrinsic tokenization metrics
│
├── models/                        # Neural backbones
│   ├── __init__.py
│   ├── TransformerCommon.py       #   Shared Plan A backbone (FIXED)
│   ├── DLinear.py                 #   Baseline (Zeng et al., 2023)
│   ├── PatchTST.py                #   Baseline (Nie et al., 2023)
│   ├── TimeMixer.py               #   Baseline (S. Wang et al., 2024)
│   └── TimeXer.py                 #   Baseline (Y. Wang et al., 2024)
│
├── layers/                        # Shared network components
│   ├── __init__.py
│   ├── StandardNorm.py            #   RevIN (Kim et al., 2022)
│   ├── Transformer_EncDec.py      #   Encoder / EncoderLayer
│   ├── SelfAttention_Family.py    #   FullAttention, AttentionLayer
│   ├── Autoformer_EncDec.py       #   series_decomp, moving_avg
│   └── Embed.py                   #   PatchEmbedding, DataEmbedding_wo_pos, etc.
│
├── exp/                           # Experiment classes
│   ├── __init__.py
│   ├── exp_basic.py               #   Base class (device, model registry)
│   ├── exp_plan_a.py              #   Plan A / Paper-1: techniques vs HMM (univariate)
│   ├── exp_plan_b.py              #   Plan B: HMM-M (multivariate)
│   └── exp_long_term_forecasting.py  # SOTA baselines
│
├── data_provider/                 # Data loading and processing
│   ├── __init__.py
│   ├── data_factory.py            #   Factory: ETTh1/h2, Weather, ECL,
│   │                              #            Traffic, Exchange, custom
│   └── data_loader.py             #   Dataset classes + StandardScaler
│
├── utils/                         # General utilities
│   ├── __init__.py
│   ├── metrics.py                 #   MSE, MAE, RMSE, MAPE, MSPE, CORR
│   ├── tools.py                   #   EarlyStopping, adjust_learning_rate, visual
│   ├── revin.py                   #   RevINNormalizer (alternative)
│   ├── timefeatures.py            #   Temporal encoding
│   ├── augmentation.py            #   Data augmentation
│   ├── masking.py                 #   Masks for attention / imputation
│   ├── dtw_metric.py              #   Dynamic Time Warping
│   └── print_args.py              #   Pretty-print of arguments
│
├── notebooks/                     # Pipeline + Paper-1 + Plan B notebooks
│   ├── pipeline_RITMO_etth2.ipynb #   4-phase pipeline validation
│   ├── eda_datasets.ipynb         #   EDA of the 6 datasets
│   ├── eda_datasets.py            #   jupytext pair of eda_datasets.ipynb
│   ├── paper1_runner.ipynb        #   Phase 2-5 launcher (Paper-1)
│   ├── paper1_final_results.ipynb #   Downstream multi-seed aggregation (Phase 3)
│   ├── paper1_intrinsic_results.ipynb # Intrinsic metrics aggregation (Phase 5)
│   ├── paper1_figures.{ipynb,py}  #   Paper draft figures
│   ├── k_sweep_plan_b.ipynb       #   K-sweep RITMO-M (multivariate HMM)
│   ├── final_results_plan_b.ipynb #   Plan B vs in-domain SOTA baselines
│   ├── fase{1..4}_*_etth2.svg     #   RITMO pipeline figures
│   └── figures_eda/               #   EDA figures (.svg)
│
├── scripts/
│   ├── long_term_forecast/        # SOTA baselines (.sh + wrapper)
│   │   ├── ETT_script/            #   8 .sh (4 baselines x ETTh1/ETTh2)
│   │   ├── ECL_script/            #   4 .sh (4 baselines x Electricity)
│   │   ├── Weather_script/        #   4 .sh
│   │   ├── Traffic_script/        #   3 .sh (cross-domain)
│   │   ├── Exchange_script/       #   1 .sh (cross-domain)
│   │   └── run_all_baselines.py   #   Resumable wrapper (--skip-cross-domain)
│   ├── paper1/                    # 6-phase multi-seed pipeline
│   │   ├── config.py              #   SEEDS, K_VALUES, DATASETS, hmm_cache_path
│   │   ├── k_optimal.json         #   Optimal K per dataset after Phase 2
│   │   ├── phase1_train_hmm_seeds.py
│   │   ├── phase2_k_sweep_seeds.py
│   │   ├── phase3_plan_a_seeds.py
│   │   ├── phase4_cross_domain_seeds.py
│   │   ├── phase5_intrinsic_metrics.py
│   │   ├── phase6_ablation.py
│   │   └── verify_hmm_convergence.py
│   └── plan_b/                    # Multivariate HMM
│       ├── plan_b_config.py
│       ├── train_hmm_caches.py    #   HMM-M caches
│       └── run_ritmo_sweep.py     #   K-sweep RITMO-M downstream
│
├── cache/                         # 122 HMM caches
│   ├── hmm_{ds}_K{k}_seed{s}.pth  #   Univariate Paper-1 (4 ds x 8 K x 3 seeds)
│   └── hmm_M_{ds}_K{k}.pth        #   Multivariate Plan B
│
├── results/                       # Per-experiment metrics (697 dirs)
│   ├── plan_a_*_ksweep_paper1_*_seed*_0/   # Phase 2 (192)
│   ├── plan_a_*_final_paper1_*_seed*_0/    # Phase 3 (288)
│   ├── plan_a_*_crossdom_paper1_*_seed*_0/ # Phase 4 (216)
│   └── paper1_intrinsic/          #   Phase 5 (12 JSONs)
│
├── test_results/                  # Per-sample visualization PDFs (gitignored)
├── checkpoints/                   # Trained Transformer weights (gitignored)
├── logs/                          # Execution logs (gitignored)
│
├── tutorial/                      # Original TSLib tutorial (reference)
│   ├── TimesNet_tutorial.ipynb
│   └── {conv,dataset,fft,result}.png
│
├── pic/                           # README images
│   ├── Pipeline-RITMO.png
│   └── notebookLM.md
│
├── run.py                         # Main entry point (unified TSLib CLI)
├── environment.yml                # Conda environment (USE THIS ONE)
├── .gitignore
└── README.md
```

## Installation

Requirements: Python 3.10, Conda.

```bash
conda env create -f environment.yml
conda activate ritmo
```

Verification:

```bash
python -c "import torch; print(torch.__version__)"
python -c "from hmm import baum_welch, viterbi_decode; print('HMM OK')"
python -c "from embeddings import EmbeddingGenerator; print('Embeddings OK')"
python -c "from models import TransformerCommon, DLinear, PatchTST; print('Models OK')"
```

## Datasets

Download from [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) and place under `./dataset/`.

| Dataset | Domain | Frequency | Observations | Role |
|---------|--------|-----------|--------------|------|
| ETTh1 | Energy | Hourly | 17,420 | HMM training |
| ETTh2 | Energy | Hourly | 17,420 | HMM training |
| Weather | Meteorology | 10 min | 52,696 | HMM training |
| Electricity | Energy | Hourly | 26,304 | HMM training |
| Traffic | Transport | Hourly | 17,544 | Zero-shot |
| Exchange | Finance | Daily | 7,588 | Zero-shot |

## Usage

### 1. Train HMM cache

```python
from hmm import baum_welch
from hmm.checkpoint import save_hmm_params
import pandas as pd, numpy as np

df = pd.read_csv('dataset/ETT-small/ETTh1.csv')
obs = df['OT'].values.astype(np.float64)
obs_norm = (obs - obs.mean()) / obs.std()

result = baum_welch(obs_norm, K=5, max_iter=100, epsilon=1e-4)
save_hmm_params(result, 'cache/hmm_etth1_K5.pth')
```

### 2. Plan A: controlled comparison

```bash
python run.py \
  --task_name plan_a \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --model_id ETTh1_96_96 --model TransformerCommon --data ETTh1 \
  --features S --seq_len 96 --pred_len 96 \
  --enc_in 1 --dec_in 1 --c_out 1 \
  --d_model 64 --n_heads 4 --e_layers 2 --d_ff 128 \
  --dropout 0.1 --batch_size 32 \
  --learning_rate 0.001 --lradj cosine --train_epochs 30 --patience 7 \
  --technique hmm_soft_residual --hmm_k 5 \
  --des plan_a_hmm_soft_res_K5 --itr 1
```

Available techniques via `--technique`:

- Baselines: `discretization`, `text_based`, `patching`, `decomposition`, `foundation`
- HMM: `hmm` (hard Viterbi), `hmm_soft`, `hmm_soft_residual`, `hmm_augmented`, `hmm_split`, `hmm_patched`

### 3. Plan B: full state-of-the-art baselines

```bash
bash scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh
bash scripts/long_term_forecast/ETT_script/DLinear_ETTh1.sh
bash scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1.sh
bash scripts/long_term_forecast/Weather_script/TimeMixer.sh
```

### 4. Validate pipeline and explore results

```bash
jupyter notebook notebooks/pipeline_RITMO_etth2.ipynb     # 4-phase validation
jupyter notebook notebooks/paper1_runner.ipynb            # Paper-1 phase launcher
jupyter notebook notebooks/paper1_final_results.ipynb     # Downstream aggregation
jupyter notebook notebooks/paper1_intrinsic_results.ipynb # Intrinsic metrics
```

## Experimental configuration

- Input: I = 96 timesteps
- Horizons: O = {96, 192, 336, 720}
- Metrics: MSE, MAE (reported), RMSE/MAPE/MSPE available
- Mode: `features S` (univariate, `OT` column in ETT / `MT_320` in ECL)
- Split: 7:1:2 (train/val/test), TSLib style
- Fixed Transformer config: `d_model=64, n_heads=4, e_layers=2, d_ff=128, dropout=0.1`
- Optimizer: Adam, `lr=1e-3`, cosine annealing
- Epochs: 30, patience: 7, batch_size: 32
- Optimal HMM K per dataset: Weather K=8, ETTh1/ETTh2 K=5

## Implemented HMM variants

Each variant in `EmbeddingGenerator` (`embeddings/embedding_generator.py`):

| Variant | Input | Formula | When to use |
|---------|-------|---------|-------------|
| `hmm` | Viterbi states | lookup `e_k = [mu_k, sigma_k, A[k,:]]` | Hard baseline |
| `hmm_soft` | gamma [T,K] | `gamma @ embedding_table` | Pure regimes (Weather) |
| `hmm_soft_residual` | gamma + x | `[r_t, mu_soft, sigma_soft, A_soft]` with `r_t=(x-mu)/sigma` | High-frequency series (ETT) |
| `hmm_augmented` | gamma + x | `[x_t, gamma_t]` | HMM enriches without replacing |
| `hmm_split` | gamma + x | Separate Linear for x and gamma + LayerNorm | Decomposition ablation |
| `hmm_patched` | gamma + x | Patches of `[x, gamma]` projected | Combines HMM + patching |

## Optimizations

- **Vectorized forward-backward**: removed the `O(T*K^2)` Python loop, ~15x speedup at T=8640.
- **Vectorized Viterbi**: removed the `O(T*K)` loop.
- **Batched forward-backward**: `forward_backward_batch()` processes `[B, T]` in a single call.
- **Verification**: bit-for-bit diff vs naive implementation (T=8640, K=5 → diff=0.0).

## Codebase

Built on top of [Time-Series-Library](https://github.com/thuml/Time-Series-Library) (THUML). The `hmm/`, `embeddings/`, `tecnicas/`, `exp/exp_plan_a.py` and `models/TransformerCommon.py` modules are original work. The baselines (DLinear, PatchTST, TimeMixer, TimeXer) and the `layers/`, `data_provider/`, `utils/` infrastructure come from the original repository.

## Key references

- Rabiner, L. R. (1989). *A tutorial on hidden Markov models*. Proc. IEEE 77(2).
- Kim, T. et al. (2022). *Reversible instance normalization for accurate time-series forecasting (RevIN)*. ICLR.
- Nie, Y. et al. (2023). *A time series is worth 64 words: long-term forecasting with transformers (PatchTST)*. ICLR.
- Zeng, A. et al. (2023). *Are transformers effective for time series forecasting? (DLinear)*. AAAI.
- Wu, H. et al. (2021). *Autoformer: decomposition transformers with auto-correlation*. NeurIPS.
- Wang, S. et al. (2024). *TimeMixer: decomposable multiscale mixing for time series forecasting*. ICLR.
- Gruver, N. et al. (2023). *Large language models are zero-shot time series forecasters (LLMTime)*. NeurIPS.
- Goswami, M. et al. (2024). *MOMENT: a family of open time-series foundation models*. ICML.
- Lin, J. et al. (2007). *Experiencing SAX: a novel symbolic representation of time series*. DMKD.

Reference PDFs under `referencias/` (7 categories, 45+ papers).
