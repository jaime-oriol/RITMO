# RITMO

**Regimenes latentes mediante Inferencia Temporal con Markov Oculto**

Tokenizacion de series temporales mediante Hidden Markov Models y evaluacion frente a tecnicas deterministas actuales (patching, decomposition, foundation models) bajo una arquitectura Transformer compartida.

Trabajo de Fin de Grado — Business Analytics, Universidad Francisco de Vitoria.

Autor: Jaime Oriol Goicoechea.

## Pregunta de investigacion

¿Pueden los estados ocultos de un HMM actuar como embeddings latentes estructurados que capturen regimenes estadisticos de forma mas efectiva que las tecnicas deterministicas actuales de tokenizacion para prediccion de series temporales univariadas?

## Contribucion

1. **HMM como tokenizador probabilistico**: los estados ocultos se proyectan a embeddings `e_k = [mu_k, sigma_k, A[k,:]]` interpretables (centro del regimen, volatilidad, dinamicas de transicion).
2. **Seis variantes de embedding HMM**: hard (Viterbi), soft (gamma posteriors), soft-residual (residual intra-regimen), augmented, split, patched.
3. **Marco de comparacion controlado (Plan A)**: mismo Transformer, misma normalizacion RevIN externa, mismos splits, mismo optimizador — solo cambia la tokenizacion.
4. **Implementacion HMM propia y vectorizada**: Baum-Welch, forward-backward batch y Viterbi en NumPy con eliminacion de loops Python (verificado bit-a-bit vs referencia naive, diff=0.0).

## Pipeline RITMO

<p align="center">
<img src="./pic/Pipeline-RITMO.png" alt="Pipeline RITMO" width="700"/>
</p>

```
Serie x --> RevIN --> HMM (Baum-Welch, K estados) --> Viterbi / Forward-Backward
                           |
                           v
                    Embeddings estructurados [mu_k, sigma_k, A[k,:]]
                           |
                           v
                    Transformer encoder (arquitectura fija)
                           |
                           v
                    RevIN inverso --> Prediccion y_hat
```

1. **RevIN**: normalizacion reversible por instancia, externa y compartida (Kim et al., 2022).
2. **Baum-Welch**: entrenamiento HMM con K estados y emisiones gaussianas. Inicializacion k-means + EM hasta convergencia (log-likelihood, epsilon=1e-4).
3. **Viterbi / Forward-Backward**: asignacion hard o posteriors gamma soft.
4. **Embeddings**: e_k por estado + variantes (soft, soft-residual, augmented, split, patched) en `embeddings/embedding_generator.py`.
5. **Transformer**: encoder con arquitectura FIJA comun a todas las tecnicas (`models/TransformerCommon.py`).

## Resultados principales

Comparacion justa Plan A (misma config `d_model=64, n_heads=4, e_layers=2, d_ff=128`, cosine LR, 30 epochs):

**Weather, pred_len=96 — HMM gana a todas las baselines:**

| # | Tecnica | MSE | MAE |
|---|---------|-----|-----|
| 1 | **hmm_soft K=8** | **0.001200** | **0.0254** |
| 2 | hmm_soft K=5 | 0.001276 | 0.0263 |
| 3 | foundation | 0.001302 | 0.0262 |
| 4 | decomposition | 0.001309 | 0.0266 |
| 5 | patching | 0.001314 | 0.0262 |

HMM supera a patching en **-8.7%** MSE. El optimo K=8 es un sweet spot claro (K=5,6,7,9,10,12 todas peores).

**ETTh1, ETTh2 y otros horizontes (resumen):**

| Dataset | Horizonte | Mejor HMM | MSE HMM | Mejor baseline | MSE baseline | Gap |
|---------|-----------|-----------|---------|----------------|--------------|-----|
| Weather | 96  | hmm_soft K=8        | 0.00120 | patching | 0.00131 | **-8.7%** |
| ETTh2   | 96  | hmm_soft_res K=5    | 0.1475  | patching | 0.1459  | +1.1% |
| ETTh1   | 96  | hmm_soft_res K=5    | 0.0583  | patching | 0.0560  | +4.2% |
| ETTh1   | 192 | hmm_soft K=5        | 0.0762  | foundation | 0.0743 | +2.6% |
| ETTh1   | 336 | hmm_soft K=5        | 0.0882  | foundation | 0.0855 | +3.2% |

En los 5 escenarios, HMM supera siempre a discretization y text_based. En horizontes largos (pl=192, pl=336) HMM supera a decomposition.

## Estructura del repositorio

```
RITMO/
├── hmm/                     # Modulo HMM (implementacion propia)
│   ├── baum_welch.py        #   Entrenamiento EM vectorizado
│   ├── forward_backward.py  #   Forward-backward + version batch
│   ├── viterbi.py           #   Decodificacion Viterbi vectorizada
│   ├── gaussian_emissions.py#   Emisiones gaussianas en log-space
│   ├── checkpoint.py        #   Guardado/carga parametros (.pth)
│   └── utils.py             #   log-normalize, init k-means
│
├── embeddings/              # Generacion de embeddings
│   └── embedding_generator.py  # 6 variantes: hard, soft, soft_residual,
│                               # augmented, split, patched
│
├── tecnicas/                # 5 tecnicas baseline de tokenizacion
│   ├── discretization.py    #   SAX (Lin et al., 2007)
│   ├── text_based.py        #   LLMTime (Gruver et al., 2023)
│   ├── patching.py          #   PatchTST (Nie et al., 2023)
│   ├── decomposition.py     #   Autoformer/DLinear (Wu et al., 2021)
│   ├── foundation.py        #   MOMENT (Goswami et al., 2024)
│   └── metrics.py           #   Metricas intrinsecas de tokenizacion
│
├── models/                  # Backbones
│   ├── TransformerCommon.py #   Backbone compartido Plan A (FIJO)
│   ├── DLinear.py           #   Baseline (Zeng et al., 2023)
│   ├── PatchTST.py          #   Baseline (Nie et al., 2023)
│   ├── TimeMixer.py         #   Baseline (S. Wang et al., 2024)
│   └── TimeXer.py           #   Baseline (Y. Wang et al., 2024)
│
├── layers/                  # Componentes compartidos
│   ├── StandardNorm.py      #   RevIN (Kim et al., 2022)
│   ├── Transformer_EncDec.py
│   ├── SelfAttention_Family.py
│   └── Embed.py
│
├── exp/                     # Clases de experimentacion
│   ├── exp_plan_a.py        #   Plan A: comparacion 6+ tecnicas
│   ├── exp_long_term_forecasting.py
│   └── exp_basic.py
│
├── data_provider/           # Carga de datos
│   ├── data_factory.py      #   Factory con ETTh1/h2, Weather, ECL, Traffic, Exchange
│   └── data_loader.py
│
├── notebooks/               # Notebooks de experimentacion
│   ├── pipeline_RITMO_etth2.ipynb   # Validacion 4 fases
│   ├── k_sweep.ipynb                # Barrido K por dataset
│   └── visualizations.ipynb         # Agregacion de resultados
│
├── scripts/                 # Scripts de ejecucion
│   ├── plan_a/              #   Scripts Plan A
│   └── long_term_forecast/  #   Scripts baselines (Plan B)
│
├── cache/                   # HMM entrenados: hmm_{dataset}_K{k}.pth
├── results/                 # metrics.npy, pred.npy, true.npy por experimento
├── test_results/            # PDFs de visualizacion de predicciones
├── utils/                   # Metricas (MSE, MAE, CORR), EarlyStopping, RevIN
├── md/                      # Documentacion (Anteproyecto, SOTA, apuntes)
├── pic/                     # Imagenes README
├── dataset/                 # Datasets (no incluidos, descargar aparte)
│
├── run.py                   # Entry point principal
├── environment.yml          # Entorno Conda
└── README.md
```

## Instalacion

Requisitos: Python 3.10, Conda.

```bash
conda env create -f environment.yml
conda activate ritmo
```

Verificacion:

```bash
python -c "import torch; print(torch.__version__)"
python -c "from hmm import baum_welch, viterbi_decode; print('HMM OK')"
python -c "from embeddings import EmbeddingGenerator; print('Embeddings OK')"
python -c "from models import TransformerCommon, DLinear, PatchTST; print('Modelos OK')"
```

## Datasets

Descargar desde [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) y colocar en `./dataset/`.

| Dataset | Dominio | Frecuencia | Observaciones | Rol |
|---------|---------|------------|---------------|-----|
| ETTh1 | Energia | Horaria | 17.420 | Entrenamiento HMM |
| ETTh2 | Energia | Horaria | 17.420 | Entrenamiento HMM |
| Weather | Meteorologia | 10 min | 52.696 | Entrenamiento HMM |
| Electricity | Energia | Horaria | 26.304 | Entrenamiento HMM |
| Traffic | Transporte | Horaria | 17.544 | Zero-shot |
| Exchange | Finanzas | Diaria | 7.588 | Zero-shot |

## Uso

### 1. Entrenar cache HMM

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

### 2. Plan A: comparacion controlada

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

Tecnicas disponibles via `--technique`:

- Baselines: `discretization`, `text_based`, `patching`, `decomposition`, `foundation`
- HMM: `hmm` (Viterbi hard), `hmm_soft`, `hmm_soft_residual`, `hmm_augmented`, `hmm_split`, `hmm_patched`

### 3. Plan B: baselines completos del estado del arte

```bash
bash scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh
bash scripts/long_term_forecast/ETT_script/DLinear_ETTh1.sh
bash scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1.sh
bash scripts/long_term_forecast/Weather_script/TimeMixer.sh
```

### 4. Validar pipeline y explorar resultados

```bash
jupyter notebook notebooks/pipeline_RITMO_etth2.ipynb  # Validacion 4 fases
jupyter notebook notebooks/k_sweep.ipynb               # Barrido K
jupyter notebook notebooks/visualizations.ipynb        # Agregacion resultados
```

## Configuracion experimental

- Input: I = 96 timesteps
- Horizontes: O = {96, 192, 336}
- Metricas: MSE, MAE (reportadas), RMSE/MAPE/MSPE disponibles
- Modo: `features S` (univariado, columna `OT` en ETT / `MT_320` en ECL)
- Split: 7:1:2 (train/val/test) estilo TSLib
- Config Transformer fija: `d_model=64, n_heads=4, e_layers=2, d_ff=128, dropout=0.1`
- Optimizador: Adam, `lr=1e-3`, cosine annealing
- Epochs: 30, patience: 7, batch_size: 32
- K HMM optimo por dataset: Weather K=8, ETTh1/ETTh2 K=5

## Variantes HMM implementadas

Cada variante en `EmbeddingGenerator` (`embeddings/embedding_generator.py`):

| Variante | Input | Formula | Cuando usar |
|----------|-------|---------|-------------|
| `hmm` | Viterbi states | lookup `e_k = [mu_k, sigma_k, A[k,:]]` | Hard baseline |
| `hmm_soft` | gamma [T,K] | `gamma @ embedding_table` | Regimenes puros (Weather) |
| `hmm_soft_residual` | gamma + x | `[r_t, mu_soft, sigma_soft, A_soft]` con `r_t=(x-mu)/sigma` | Series con alta frecuencia (ETT) |
| `hmm_augmented` | gamma + x | `[x_t, gamma_t]` | HMM enriquece sin reemplazar |
| `hmm_split` | gamma + x | Linear separado x y gamma + LayerNorm | Ablacion de decomposition |
| `hmm_patched` | gamma + x | Parches de `[x, gamma]` proyectados | Combina HMM + patching |

## Optimizaciones

- **Forward-backward vectorizado**: eliminado el loop Python `O(T*K^2)`, ~15x speedup en T=8640.
- **Viterbi vectorizado**: eliminado el loop `O(T*K)`.
- **Forward-backward batch**: `forward_backward_batch()` procesa `[B, T]` en una sola llamada.
- **Verificacion**: diff bit-a-bit vs implementacion naive (T=8640, K=5 → diff=0.0).

## Base de codigo

Construido sobre [Time-Series-Library](https://github.com/thuml/Time-Series-Library) (THUML). Los modulos `hmm/`, `embeddings/`, `tecnicas/`, `exp/exp_plan_a.py` y `models/TransformerCommon.py` son implementacion propia del TFG. Los baselines (DLinear, PatchTST, TimeMixer, TimeXer) y la infraestructura de `layers/`, `data_provider/`, `utils/` vienen del repo original.

## Referencias clave

- Rabiner, L. R. (1989). *A tutorial on hidden Markov models*. Proc. IEEE 77(2).
- Kim, T. et al. (2022). *Reversible instance normalization for accurate time-series forecasting (RevIN)*. ICLR.
- Nie, Y. et al. (2023). *A time series is worth 64 words: long-term forecasting with transformers (PatchTST)*. ICLR.
- Zeng, A. et al. (2023). *Are transformers effective for time series forecasting? (DLinear)*. AAAI.
- Wu, H. et al. (2021). *Autoformer: decomposition transformers with auto-correlation*. NeurIPS.
- Wang, S. et al. (2024). *TimeMixer: decomposable multiscale mixing for time series forecasting*. ICLR.
- Gruver, N. et al. (2023). *Large language models are zero-shot time series forecasters (LLMTime)*. NeurIPS.
- Goswami, M. et al. (2024). *MOMENT: a family of open time-series foundation models*. ICML.
- Lin, J. et al. (2007). *Experiencing SAX: a novel symbolic representation of time series*. DMKD.

Ver `md/SOTA.md` para la revision completa del estado del arte.
