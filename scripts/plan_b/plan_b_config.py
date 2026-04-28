"""Config compartida para los scripts de Plan B.

Centraliza la lista de datasets, valores de K, variantes HMM y horizontes.
Editable: cambia los valores aqui y lo aplica a train_hmm_caches.py y run_ritmo_sweep.py.
"""

# Semilla unica (fijada en los scripts y en run.py).
SEED = 2021

# Valores de K a barrer por dataset (mismo rango que Plan A para comparacion directa).
K_VALUES_BY_DATASET = {
    'ETTh1':       [3, 4, 5, 6, 7, 8, 9, 10],
    'ETTh2':       [3, 4, 5, 6, 7, 8, 9, 10],
    'Weather':     [3, 4, 5, 6, 7, 8, 9, 10],
    'Electricity': [3, 4, 5, 6, 7, 8, 9, 10],
}

# Variantes HMM a evaluar (las ganadoras en Plan A).
VARIANTS = ['hmm_soft', 'hmm_soft_residual']

# Horizontes estandar TSLib.
HORIZONS = [96, 192, 336, 720]

# Metadatos por dataset. data_arg se pasa a --data (debe ser una key de data_provider.data_dict).
# train_epochs moderado para Weather/ECL por volumen; ETT conserva 30 como Plan A.
DATASETS = [
    {
        'name': 'ETTh1', 'data_arg': 'ETTh1',
        'root': './dataset/ETT-small/', 'data_path': 'ETTh1.csv',
        'csv': 'dataset/ETT-small/ETTh1.csv',
        'C': 7, 'freq': 'h', 'batch_size': 32, 'train_epochs': 30,
        'split_type': 'ETT', 'hmm_chunk_size': 32,
    },
    {
        'name': 'ETTh2', 'data_arg': 'ETTh2',
        'root': './dataset/ETT-small/', 'data_path': 'ETTh2.csv',
        'csv': 'dataset/ETT-small/ETTh2.csv',
        'C': 7, 'freq': 'h', 'batch_size': 32, 'train_epochs': 30,
        'split_type': 'ETT', 'hmm_chunk_size': 32,
    },
    {
        'name': 'Weather', 'data_arg': 'Weather',
        'root': './dataset/weather/', 'data_path': 'weather.csv',
        'csv': 'dataset/weather/weather.csv',
        'C': 21, 'freq': 'h', 'batch_size': 32, 'train_epochs': 10,
        'split_type': 'custom', 'hmm_chunk_size': 21,
    },
    {
        'name': 'Electricity', 'data_arg': 'Electricity',
        'root': './dataset/electricity/', 'data_path': 'electricity.csv',
        'csv': 'dataset/electricity/electricity.csv',
        'C': 321, 'freq': 'h', 'batch_size': 16, 'train_epochs': 5,
        'split_type': 'custom', 'hmm_chunk_size': 128,
    },
]


def cache_path(dataset_name: str, K: int) -> str:
    """Ruta canonica de la cache HMM-M para un (dataset, K)."""
    return f"./cache/hmm_M_{dataset_name.lower()}_K{K}.pth"


def experiment_tag(dataset_name: str, variant: str, K: int, pred_len: int) -> str:
    """Identificador unico (usado como model_id y --des en run.py)."""
    return f"PLANB_{dataset_name}_{variant}_K{K}_pl{pred_len}"
