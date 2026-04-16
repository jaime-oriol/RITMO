"""Config compartida para los scripts del Paper-1.

Centraliza datasets, Ks, seeds, variantes HMM y horizontes.
Fuente de verdad: memoria/secciones/plan_paper.md
"""

# 3 seeds para todas las fases del Paper-1.
# 42 ya existe en las caches actuales (cache/hmm_*_K*.pth sin sufijo seed).
SEEDS = [42, 2021, 7]

# Barrido K: mismo rango que el TFG original.
K_VALUES = [3, 4, 5, 6, 7, 8, 9, 10]

# Variantes HMM del Paper-1 (sin 'hard', según plan).
HMM_VARIANTS = ['hmm_soft', 'hmm_soft_residual']

# Horizontes TSLib estándar.
HORIZONS = [96, 192, 336, 720]

# K óptimo por dataset tras Plan A original (seed 42) -- se re-validará con 3 seeds.
# Fuente: notebooks/final_results.ipynb y TFG_RITMO.md.
OPTIMAL_K_SEED42 = {
    'ETTh1':       {'variant': 'hmm_soft_residual', 'K': 4},
    'ETTh2':       [
        {'variant': 'hmm_soft', 'K': 6},
        {'variant': 'hmm_soft', 'K': 8},  # empatan en downstream
    ],
    'Weather':     {'variant': 'hmm_soft', 'K': 8},
    'Electricity': {'variant': 'hmm_soft_residual', 'K': 3},
}

# Datasets del Grupo 1 (in-domain).
DATASETS = [
    {
        'name': 'ETTh1', 'data_arg': 'ETTh1',
        'root': './dataset/ETT-small/', 'data_path': 'ETTh1.csv',
        'csv': 'dataset/ETT-small/ETTh1.csv',
        'target': 'OT', 'freq': 'h',
        'split_type': 'ETT', 'batch_size': 32, 'train_epochs': 30,
    },
    {
        'name': 'ETTh2', 'data_arg': 'ETTh2',
        'root': './dataset/ETT-small/', 'data_path': 'ETTh2.csv',
        'csv': 'dataset/ETT-small/ETTh2.csv',
        'target': 'OT', 'freq': 'h',
        'split_type': 'ETT', 'batch_size': 32, 'train_epochs': 30,
    },
    {
        'name': 'Weather', 'data_arg': 'Weather',
        'root': './dataset/weather/', 'data_path': 'weather.csv',
        'csv': 'dataset/weather/weather.csv',
        'target': 'OT', 'freq': 'h',
        'split_type': 'custom', 'batch_size': 32, 'train_epochs': 30,
    },
    {
        'name': 'Electricity', 'data_arg': 'custom',  # usa Dataset_Custom
        'root': './dataset/electricity/', 'data_path': 'electricity.csv',
        'csv': 'dataset/electricity/electricity.csv',
        'target': 'OT', 'freq': 'h',
        'split_type': 'custom', 'batch_size': 32, 'train_epochs': 30,
    },
]

# Datasets del Grupo 2 (cross-domain, solo para fase 4).
CROSS_DOMAIN_TARGETS = [
    {
        'name': 'Traffic', 'data_arg': 'custom',
        'root': './dataset/traffic/', 'data_path': 'traffic.csv',
        'target': 'OT', 'freq': 'h',
        'batch_size': 32, 'train_epochs': 10, 'patience': 3,
    },
    {
        'name': 'Exchange', 'data_arg': 'custom',
        'root': './dataset/exchange_rate/', 'data_path': 'exchange_rate.csv',
        'target': 'OT', 'freq': 'd',
        'batch_size': 32, 'train_epochs': 10, 'patience': 3,
    },
]

# Cross-domain: horizontes reducidos (como en zero_shot.ipynb).
CROSS_DOMAIN_HORIZONS = [96, 336]

# 5 técnicas baseline.
BASELINE_TECHNIQUES = ['discretization', 'text_based', 'patching', 'decomposition', 'foundation']

# Transformer config (fija, idéntica al Plan A original).
TRANSFORMER_CFG = {
    'seq_len': 96, 'label_len': 48,
    'd_model': 64, 'n_heads': 4, 'e_layers': 2, 'd_ff': 128,
    'dropout': 0.1, 'batch_size': 32, 'learning_rate': 0.001,
    'lradj': 'cosine', 'train_epochs': 30, 'patience': 7,
}


def hmm_cache_path(dataset_name: str, K: int, seed: int) -> str:
    """Ruta canónica de cache HMM univariante con seed explícita.

    Naming: cache/hmm_{dataset_lower}_K{k}_seed{s}.pth
    La cache sin sufijo seed (cache/hmm_{ds}_K{k}.pth) corresponde a seed=42.
    """
    ds_key = dataset_name.lower()
    # Electricity usa 'custom' como data_arg pero el cache se llama 'electricity' para claridad.
    return f"./cache/hmm_{ds_key}_K{K}_seed{seed}.pth"


def hmm_cache_path_legacy(dataset_name: str, K: int) -> str:
    """Cache legacy sin sufijo seed (seed=42 implícita). Usada por scripts previos."""
    ds_key = dataset_name.lower()
    # Note: Electricity era 'custom' en los caches legacy del TFG.
    if dataset_name == 'Electricity':
        return f"./cache/hmm_custom_K{K}.pth"
    return f"./cache/hmm_{ds_key}_K{K}.pth"
