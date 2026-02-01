# CLAUDE.md - Guía de Desarrollo RITMO

## Filosofía Core

**Ve paso a paso, uno a uno. Despacio es el camino más rápido. Escribe siempre el código lo más compacto y conciso posible, y que cumpla exactamente lo pedido al 100%. Sin emojis ni florituras. Usa nombres claros y estándar. Incluye solo comentarios útiles y necesarios.**

Antes de realizar cualquier tarea, revisa cuidadosamente el archivo CLAUDE.md.
Aquí encontrarás las directrices de trabajo y la estructura del proyecto que debes seguir.

source ~/anaconda3/etc/profile.d/conda.sh && conda activate ritmo && python << 'EOF'

### Principios de Desarrollo

- **KISS (Keep It Simple, Stupid)**: Elige soluciones simples sobre complejas
- **YAGNI (You Aren't Gonna Need It)**: Implementa características solo cuando sean necesarias
- **Fail Fast**: Detecta errores temprano y lanza excepciones inmediatamente
- **Single Responsibility**: Cada función, clase y módulo tiene un propósito claro
- **Dependency Inversion**: Módulos de alto nivel dependen de abstracciones, no implementaciones

## Descripción del Proyecto

**RITMO** - **R**egímenes latentes mediante **I**nferencia **T**emporal con **M**arkov **O**culto para tokenización y forecasting de series temporales

### Objetivo del TFG

Desarrollar e implementar un sistema de tokenización de series temporales basado en Estados Ocultos de Markov (HMM), donde los estados ocultos actúen como embeddings latentes estructurados, evaluar su desempeño en tareas de predicción a largo plazo frente a técnicas determinísticas actuales (PatchTST, DLinear, TimeMixer, TimeXer), y analizar los trade-offs entre complejidad computacional, ratio de compresión e interpretabilidad de regímenes.

### Pregunta de Investigación

¿Pueden los estados ocultos de un Hidden Markov Model actuar como embeddings latentes estructurados que capturen dependencias temporales y regímenes estadísticos de manera más efectiva que las técnicas determinísticas actuales de tokenización para la predicción de series temporales univariadas en el contexto de modelos de lenguaje?

Para más detalles, consulta `Anteproyecto-RITMO.md`.

## Estructura del Repositorio

```
RITMO/
├── models/                  # Modelos baseline del TFG (4 modelos)
│   ├── __init__.py         # Inicialización del módulo
│   ├── DLinear.py          # Baseline 1: Descomposición + Linear
│   ├── PatchTST.py         # Baseline 2: Patch-based Transformer
│   ├── TimeMixer.py        # Baseline 3: Multi-scale mixing
│   └── TimeXer.py          # Baseline 4: Exogenous variables
│
├── layers/                  # Componentes compartidos de redes neuronales
│   ├── Embed.py            # PatchEmbedding, DataEmbedding_wo_pos, etc.
│   ├── Transformer_EncDec.py # Encoder, EncoderLayer
│   ├── SelfAttention_Family.py # FullAttention, AttentionLayer
│   ├── Autoformer_EncDec.py # series_decomp, moving_avg
│   ├── StandardNorm.py     # Normalize (RevIN-style)
│   └── [otros componentes] # Más layers disponibles
│
├── exp/                     # Clases de experimentación
│   ├── exp_basic.py        # Clase base con registro de modelos
│   ├── exp_long_term_forecasting.py # Experimentos long-term
│   ├── exp_short_term_forecasting.py
│   ├── exp_imputation.py
│   ├── exp_anomaly_detection.py
│   └── exp_classification.py
│
├── data_provider/          # Carga y procesamiento de datos
│   ├── data_factory.py     # Factory pattern para datasets
│   ├── data_loader.py      # Loaders para ETT, Weather, etc.
│   ├── m4.py               # Dataset M4
│   └── uea.py              # Dataset UEA
│
├── utils/                   # Utilidades generales
│   ├── metrics.py          # MSE, MAE, RMSE, MAPE, MSPE
│   ├── tools.py            # Funciones helper
│   ├── timefeatures.py     # Codificación temporal
│   ├── augmentation.py     # Data augmentation
│   ├── revin.py            # RevINNormalizer (TFG RITMO)
│   └── losses.py           # Funciones de pérdida
│
├── hmm/                     # Módulo HMM para TFG RITMO
│   ├── __init__.py         # Exporta baum_welch, viterbi_decode
│   ├── utils.py            # log_normalize, initialize_kmeans
│   ├── gaussian_emissions.py # Emisiones gaussianas
│   ├── forward_backward.py # Algoritmo Forward-Backward (E-step)
│   ├── baum_welch.py       # Algoritmo EM para HMM
│   └── viterbi.py          # Decodificación óptima de estados
│
├── embeddings/              # Generación de embeddings estructurados
│   ├── __init__.py         # Exporta EmbeddingGenerator
│   └── embedding_generator.py # e_k = [μ_k, σ_k, A[k,:]]
│
├── tecnicas/                # Implementaciones de 5 técnicas de tokenización
│   ├── discretization.py   # Técnica: Discretización (ej: SAX)
│   ├── text_based.py       # Técnica: Text-based (ej: LLMTime)
│   ├── patching.py         # Técnica: Patching (ej: PatchTST)
│   ├── decomposition.py    # Técnica: Descomposición (ej: DLinear)
│   ├── foundation.py       # Técnica: Foundation models (ej: MOMENT)
│   ├── Electricity_tokenization.ipynb # Comparación en Electricity
│   ├── ETTh2_tokenization.ipynb       # Comparación en ETTh2
│   └── figures/            # Exportación automática de visualizaciones
│
├── notebooks/               # Notebooks de validación
│   ├── RITMO_pipeline_validation.ipynb # Validación 4 fases
│   └── figures/            # Figuras del pipeline
│
├── md/                      # Documentación técnica
│   ├── RITMO_VALIDACION.md # Validación del pipeline
│   └── TECNICAS.md         # Comparativa de 6 técnicas
│
├── cache/                   # Cache de parámetros entrenados
│   └── hmm_*.pth           # Parámetros HMM por dataset
│
├── scripts/                 # Scripts de ejecución (26 scripts)
│   ├── long_term_forecast/ # Scripts para forecasting
│   │   ├── ETT_script/     # ETTh1, ETTh2 (7 scripts)
│   │   ├── ECL_script/     # Electricity (4 scripts)
│   │   ├── Weather_script/ # Weather (3 scripts)
│   │   ├── Traffic_script/ # Traffic (3 scripts)
│   │   └── Exchange_script/# Exchange (1 script)
│   └── exogenous_forecast/ # Scripts TimeXer con variables exógenas
│       ├── ETTh1/
│       ├── ETTh2/
│       ├── ECL/
│       ├── Traffic/
│       └── Weather/
│
├── tutorial/               # Tutorial y recursos
│   └── TimesNet_Tutorial.ipynb
│
├── pic/                    # Imágenes del README
│
├── dataset/                # Datasets (no incluidos, descargar aparte)
│   ├── ETT-small/          # ETTh1, ETTh2, ETTm1, ETTm2
│   ├── weather/
│   ├── electricity/
│   ├── traffic/
│   └── exchange/
│
├── .claude/                # Configuración Claude Code
│   └── settings.local.json
├── run.py                  # Script principal de ejecución
├── requirements.txt        # Dependencias Python (DESACTUALIZADO)
├── environment.yml         # Especificación Conda (USAR ESTE)
├── Anteproyecto-RITMO.md  # Documento del TFG
├── README.md               # Información del proyecto
└── CLAUDE.md               # Esta guía de desarrollo
```

## Modelos Baseline

### 1. DLinear (Zeng et al., 2023)
**Archivo:** `models/DLinear.py`

Baseline simple que demuestra que descomposición + capas lineales puede superar a transformers complejos.

**Características:**
- Descomposición serie temporal (trend + seasonal)
- Dos capas lineales separadas
- Channel-independent modeling
- Extremadamente eficiente

**Dependencias:**
- `layers.Autoformer_EncDec.series_decomp`

### 2. PatchTST (Nie et al., 2023)
**Archivo:** `models/PatchTST.py`

Baseline principal con patch-based tokenization que reduce complejidad de O(L²) a O((L/S)²).

**Características:**
- Segmentación en patches como tokens
- Encoder-only Transformer
- Channel-independence
- Normalización Non-stationary

**Dependencias:**
- `layers.Transformer_EncDec.Encoder`, `EncoderLayer`
- `layers.SelfAttention_Family.FullAttention`, `AttentionLayer`
- `layers.Embed.PatchEmbedding`

### 3. TimeMixer (Wang et al., 2024)
**Archivo:** `models/TimeMixer.py`

Baseline con descomposición multi-escala y mixing operations.

**Características:**
- Past Decomposable Mixing (PDM)
- Multi-scale processing con down-sampling
- Soporta moving avg y DFT decomposition
- Channel independence/dependence configurable

**Dependencias:**
- `layers.Autoformer_EncDec.series_decomp`
- `layers.Embed.DataEmbedding_wo_pos`
- `layers.StandardNorm.Normalize`

### 4. TimeXer (Wang et al., 2024)
**Archivo:** `models/TimeXer.py`

Baseline para forecasting con variables exógenas.

**Características:**
- Paradigma exogenous variable forecasting
- Embeddings endógenos y exógenos
- Global token mechanism
- Cross-attention entre variables

**Dependencias:**
- `layers.SelfAttention_Family.FullAttention`, `AttentionLayer`
- `layers.Embed.DataEmbedding_inverted`, `PositionalEmbedding`

## Datasets del TFG

### Datasets de Entrenamiento HMM (4)

1. **ETTh1** - Electric Transformer Temperature (hourly)
   - 7 características, frecuencia horaria
   - Split 7:1:2 (train/val/test)
   - Benchmark universal

2. **ETTh2** - Electric Transformer Temperature (hourly)
   - Variante con distribution shift
   - Valida robustez de RevIN

3. **Weather** - Temperatura wet-bulb
   - Regímenes climáticos estacionales claros
   - Ideal para evaluar cambios de régimen

4. **Electricity (ECL)** - Consumo eléctrico MT_320
   - Periodicidad diaria fuerte (24h)
   - Patrones día/noche claros

### Datasets Zero-Shot (2)

5. **Traffic** - Ocupación sensores de tráfico
   - Dominio diferente, sin re-entrenamiento
   - Regímenes rush-hour explícitos

6. **Exchange** - Tipos de cambio
   - Series financieras sin periodicidad marcada
   - Test de robustez para HMM

**Nota:** Los datasets deben descargarse desde Google Drive o Baidu Drive según README.md original.

## Metodología del TFG

### Pipeline RITMO Propuesto

```
1. NORMALIZACIÓN (RevIN)
   └── X_norm = (X - μ) / σ
       ↓
2. ENTRENAMIENTO HMM (Baum-Welch)
   ├── Sobre datasets: ETTh1, ETTh2, Weather, Electricity
   ├── Estima parámetros λ* = (A*, B*, π*)
   └── Emisiones gaussianas N(μ_k, σ²_k)
       ↓
3. TOKENIZACIÓN (Viterbi)
   ├── Q* = argmax_Q P(Q|O, λ*)
   └── Secuencia [z₁, z₂, ..., z_T]
       ↓
4. EMBEDDINGS ESTRUCTURADOS
   ├── e_k = [μ_k, σ_k, A[k,:]]
   ├── μ_k: centro del régimen
   ├── σ_k: volatilidad del régimen
   └── A[k,:]: dinámicas de transición
       ↓
5. PREDICCIÓN (Transformer)
   ├── ŷ_norm = Transformer([e_{z₁}, ..., e_{z_I}])
   └── ŷ = ŷ_norm × σ + μ
```

### Configuración Experimental

- **Input length:** I = 96 timesteps
- **Prediction horizons:** O ∈ {96, 192, 336, 720}
- **Métricas:** MSE, MAE
- **Comparación:** 4 baselines en 6 datasets

## Módulo de Embeddings

### EmbeddingGenerator
**Archivo:** `embeddings/embedding_generator.py`

Genera embeddings estructurados desde parámetros HMM entrenados.

**Características:**
- Concatena estadísticas del régimen (μ_k, σ_k) con dinámicas de transición (A[k,:])
- Proyección lineal opcional a d_model dimensional
- Compatible con PyTorch para backpropagation end-to-end
- Embeddings interpretables vs black-box learnables

**Uso:**
```python
from embeddings import EmbeddingGenerator

hmm_params = {'A': A, 'mu': mu, 'sigma': sigma, 'pi': pi}
emb_gen = EmbeddingGenerator(hmm_params, d_model=128, device='cpu')
embedding_table = emb_gen.get_embedding_table()  # [K, d_model]
```

## Notebooks de Validación

### RITMO Pipeline Validation
**Archivo:** `notebooks/RITMO_pipeline_validation.ipynb`

Notebook de validación exhaustiva del pipeline RITMO en 4 fases:

1. **Fase 1 - RevIN**: Normalización reversible con MSE < 1e-12
2. **Fase 2 - Baum-Welch**: Convergencia EM con tracking de log-likelihoods
3. **Fase 3 - Viterbi**: Tokenización óptima con ratio compresión 27x
4. **Fase 4 - Embeddings**: Visualización embedding space μ-σ + matriz A

**Convenciones:**
- Auto-save PNG en `notebooks/` (sin PDF)
- Visualizaciones profesionales con offsets y flechas
- Cache HMM en `cache/hmm_etth1_K5.pth`

### Notebooks de Técnicas
**Archivos:** `tecnicas/Electricity_tokenization.ipynb`, `tecnicas/ETTh2_tokenization.ipynb`

Notebooks comparativos de 6 técnicas de tokenización:

1. **HMM (RITMO)** - Propuesta del TFG: estados ocultos con Viterbi
2. **Discretización** - Símbolos discretos (ej: SAX, VQ-VAE)
3. **Text-based** - Serialización a texto (ej: LLMTime)
4. **Patching** - Segmentación en patches (ej: PatchTST)
5. **Descomposición** - Trend + seasonal (ej: DLinear, Autoformer)
6. **Foundation** - Pre-entrenamiento masivo (ej: MOMENT)

**Convenciones:**
- Auto-save PNG en cada celda de visualización
- Figuras exportadas a `tecnicas/figures/`
- Nombres: `{tecnica}_{dataset}.png`

## Técnicas de Tokenización

### 1. HMM - RITMO (Propuesta del TFG)
**Archivo:** `hmm/baum_welch.py`, `hmm/viterbi.py`

- Vocabulario: K=5 estados ocultos
- Compresión: 27x vía run-length encoding
- Embeddings: [μ_k, σ_k, A[k,:]] interpretables

### 2. Discretización
**Archivo:** `tecnicas/discretization.py`

- Ejemplo: SAX (Lin et al., 2007)
- Vocabulario: Finito (ej: 8 símbolos)
- Compresión: 1x (sin compresión)
- Embeddings: Lookup table aprendible

### 3. Text-based
**Archivo:** `tecnicas/text_based.py`

- Ejemplo: LLMTime (Gruver et al., 2023)
- Vocabulario: Caracteres (0-9, signo, punto, espacio)
- Compresión: 0.1x (expansión 10x)
- Embeddings: Character embeddings

### 4. Patching
**Archivo:** `tecnicas/patching.py`

- Ejemplo: PatchTST (Nie et al., 2023)
- Vocabulario: Continuo (patches R^16)
- Compresión: 16x (patches no solapados)
- Embeddings: Proyección lineal

### 5. Descomposición
**Archivo:** `tecnicas/decomposition.py`

- Ejemplo: DLinear (Zeng et al., 2023)
- Vocabulario: 2 componentes (trend + seasonal)
- Compresión: Variable
- Embeddings: Proyección por componente

### 6. Foundation Models
**Archivo:** `tecnicas/foundation.py`

- Ejemplo: MOMENT (Goswami et al., 2024)
- Vocabulario: Patches con masking 30%
- Compresión: 16x
- Embeddings: Patch + position + mask token

## Convenciones de Visualización

### Estándares Profesionales

**Matplotlib setup:**
```python
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
```

**Paleta de colores:**
- Usar Okabe-Ito (colorblind-friendly)
- `colors_oi = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']`

**Elementos obligatorios:**
- Títulos con `fontweight='bold'`
- Grids con `alpha=0.3`
- `tight_layout()` antes de guardar
- Colorbars con labels rotados
- Anotaciones con bbox para contraste

**Auto-save de figuras:**
```python
fig.savefig('figures/{nombre}_{dataset}.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Visualizaciones Específicas

**Embedding space:**
- Scatter μ-σ con círculos proporcionales a volatilidad
- Anotaciones con offsets estratégicos y flechas
- Matriz de transiciones con valores anotados

**Convergencia HMM:**
- Log-likelihood en eje Y izquierdo
- |ΔLL| en eje Y derecho (escala log)
- Threshold ε=1e-4 marcado

**Tokenización:**
- Serie con tokens coloreados
- Barra compacta de run-length encoding
- Distribución de frecuencias

## Estándares de Desarrollo

### Estilo de Código

```python
# Convenciones de nombres
nombre_variable = "ejemplo"       # snake_case para variables/funciones
class GestorUsuario:              # PascalCase para clases
MAX_REINTENTOS = 3                # UPPER_CASE para constantes
_metodo_interno()                 # Guión bajo para privados

# Type hints requeridos
def procesar_datos(datos: List[Dict]) -> pd.DataFrame:
    """Procesar con tipos claros."""

# Docstrings obligatorios
def extraer_datos(liga: str, temporada: str) -> Dict[str, Any]:
    """
    Extrae datos de la fuente.

    Args:
        liga: Identificador de liga (ej: 'ESP-La Liga')
        temporada: Temporada en formato YY-YY (ej: '23-24')

    Returns:
        Diccionario con datos extraídos

    Raises:
        ValueError: Si liga o temporada inválidas
    """
```

### Manejo de Errores

```python
# Manejo específico de excepciones
try:
    datos = scraper.extraer()
except ConnectionError as e:
    logger.error(f"Error de red: {e}")
    return datos_cacheados
except ValueError as e:
    logger.error(f"Validación de datos falló: {e}")
    raise
```

## Gestión del Entorno

### Entorno Conda 'ritmo'

**Instalación inicial:**

```bash
# Opción 1: Desde environment.yml (recomendado)
conda env create -f environment.yml

# Opción 2: Manual
conda create -n ritmo python=3.10 -y
conda activate ritmo
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt  # Después de actualizar versiones
```

**Workflow diario:**

```bash
# Activar el entorno
conda activate ritmo

# Verificar instalación
python -c "import torch; print(torch.__version__)"
python -c "from models import DLinear, PatchTST, TimeMixer, TimeXer"

# Trabajar en el proyecto
python run.py --model PatchTST --data ETTh1 --task_name long_term_forecast

# Desactivar al terminar
conda deactivate
```

**Gestión del entorno:**

```bash
# Ver paquetes instalados
conda list

# Actualizar entorno tras cambios
conda env update -f environment.yml --prune

# Exportar configuración actual
conda env export > environment_backup.yml

# Eliminar entorno
conda env remove -n ritmo
```

### Dependencias Principales

**PyTorch y Científico:**
- torch==2.9.0+cpu (PyTorch 2.x CPU)
- numpy>=2.1.2
- pandas>=2.3.3
- scikit-learn>=1.7.2
- scipy>=1.15.3
- matplotlib>=3.10.7

**Time Series Específico:**
- einops==0.8.0
- local-attention==1.9.14
- reformer-pytorch==1.4.4
- sktime>=0.39.0
- PyWavelets>=1.8.0

**Utilidades:**
- tqdm>=4.67.1
- patool>=4.0.1

**Nota:** El archivo `requirements.txt` original tiene versiones antiguas (torch==1.7.1). Usar `environment.yml` que tiene versiones actualizadas.

## Ejecución de Experimentos

### Script Principal (run.py)

```bash
# Ejecución básica
python run.py \
  --model PatchTST \
  --data ETTh1 \
  --task_name long_term_forecast \
  --seq_len 96 \
  --pred_len 96 \
  --features S

# Parámetros principales
--model: DLinear | PatchTST | TimeMixer | TimeXer
--data: ETTh1 | ETTh2 | Weather | Electricity | Traffic | Exchange
--task_name: long_term_forecast (otros: short_term, imputation, etc.)
--seq_len: Longitud de entrada (default: 96)
--pred_len: Horizonte de predicción (96 | 192 | 336 | 720)
--features: S (univariate) | M (multivariate) | MS (multi-to-uni)

# GPU
--use_gpu: True | False
--gpu: 0 (número de GPU)
--use_multi_gpu: Para múltiples GPUs
```

### Scripts de Shell

```bash
# Ejecutar script de dataset específico
bash scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh
bash scripts/long_term_forecast/Weather_script/TimeMixer.sh

# Los scripts ya tienen configuración óptima:
# - Input length: 96
# - Prediction lengths: 96, 192, 336, 720
# - Features: S (univariate para TFG)
# - Múltiples experimentos secuenciales
```

## Workflow de Git

### Estrategia de Ramas

```
main (rama protegida)
  ├── feature/hmm-tokenization
  ├── feature/embedding-generation
  ├── fix/viterbi-optimization
  └── docs/update-methodology
```

### Convenciones de Nombres de Ramas

- `feature/` - Nueva funcionalidad
- `fix/` - Corrección de bugs
- `docs/` - Actualizaciones de documentación
- `refactor/` - Mejoras de código sin cambiar funcionalidad
- `test/` - Adiciones o modificaciones de tests
- `experiment/` - Experimentos específicos del TFG

### Flujo de Trabajo

```bash
# 1. Comenzar nueva tarea - siempre desde main
git checkout main
git pull origin main
git checkout -b feature/hmm-implementation

# 2. Trabajo con commits incrementales
git add -p  # Revisar cambios pieza por pieza
git commit -m "feat: añadir estructura base para HMM"
git commit -m "feat: implementar algoritmo Baum-Welch"
git commit -m "feat: añadir Viterbi para tokenización"

# 3. Mantener rama actualizada con main
git fetch origin
git rebase origin/main

# 4. Push a remoto
git push origin feature/hmm-implementation

# 5. Después de aprobar y mergear
git checkout main
git pull origin main
git branch -d feature/hmm-implementation
```

### Formato de Mensajes de Commit

Seguir especificación de conventional commits:

```bash
# Formato: <tipo>(<ámbito>): <asunto>

# Tipos
feat: Nueva funcionalidad
fix: Corrección de bug
docs: Cambios en documentación
style: Cambios de estilo de código
refactor: Cambios que no arreglan bugs ni añaden features
perf: Mejoras de rendimiento
test: Añadir o modificar tests
chore: Tareas de mantenimiento

# Ejemplos para TFG
git commit -m "feat(hmm): implementar Baum-Welch con emisiones gaussianas"
git commit -m "fix(viterbi): corregir cálculo de probabilidades de transición"
git commit -m "docs: actualizar metodología en Anteproyecto-RITMO.md"
git commit -m "experiment: evaluar K=5 estados en ETTh1"
```

## Configuración Claude Code

### Setup Inicial

```bash
# Saltar prompts de permisos para workflow más rápido
claude --dangerously-skip-permissions

# Configurar terminal para mejor experiencia
/terminal-setup

# Limpiar chat entre diferentes tareas
/clear
```

### Mejores Prácticas

**Operaciones con Archivos:**
- Shift+drag para referenciar archivos (no drag regular)
- Control+V para pegar imágenes (no Command+V)
- Usar `@filename` para referenciar archivos específicos

**Gestión del Chat:**
- Encolar múltiples prompts para procesamiento en lote
- Escape para detener Claude (no Control+C)
- Escape dos veces para ver historial de mensajes
- Flecha arriba para navegar comandos previos

### Contexto del Proyecto (CLAUDE.md)

Los archivos CLAUDE.md proporcionan contexto de proyecto jerárquicamente:
- `CLAUDE.md` raíz - Overview del proyecto y estándares
- `README.md` de subdirectorio - Guías específicas de módulo
- Los archivos más específicos tienen precedencia

## Métricas de Evaluación

### Métricas Principales

```python
# Implementadas en utils/metrics.py

MSE  = Mean Squared Error (métrica principal)
MAE  = Mean Absolute Error (métrica principal)
RMSE = Root Mean Squared Error
MAPE = Mean Absolute Percentage Error
MSPE = Mean Squared Percentage Error
```

### Protocolo de Evaluación TFG

1. **Desempeño predictivo:**
   - MSE y MAE promediados sobre horizontes O ∈ {96, 192, 336, 720}
   - Input fijo I=96
   - Reportar por dataset y promedio general

2. **Ratio de compresión:**
   - Timesteps originales / tokens generados
   - Para HMM: número de cambios de estado (Viterbi)
   - Para PatchTST: T/P con P=16

3. **Eficiencia computacional:**
   - Tiempo de entrenamiento por epoch
   - Tiempo de inferencia por muestra
   - Throughput (muestras/segundo)

4. **Interpretabilidad de regímenes (solo HMM):**
   - Distribución temporal de activaciones
   - Interpretación de parámetros (μ_k, σ_k)
   - Estructura de transiciones en matriz A

## Referencias

### Documentación Interna

- **Anteproyecto TFG**: `Anteproyecto-RITMO.md` - Metodología completa
- **Validación Pipeline**: `md/RITMO_VALIDACION.md` - Validación 4 fases con métricas
- **Comparativa Técnicas**: `md/TECNICAS.md` - Análisis técnico de 6 técnicas
- **README Original**: `README.md` - Información Time-Series-Library
- **Environment**: `environment.yml` - Especificación completa del entorno

**Nota sobre documentación:** Los archivos en `md/` usan prosa técnica densa (1-2 líneas por concepto) en lugar de bullet points, maximizando densidad informativa manteniendo accesibilidad.

### Recursos Externos

- **Time-Series-Library**: https://github.com/thuml/Time-Series-Library
- **Papers de Modelos:**
  - PatchTST: Nie et al., 2023
  - DLinear: Zeng et al., 2023
  - TimeMixer: Wang et al., 2024
  - TimeXer: Wang et al., 2024
- **HMM Theory:**
  - Rabiner, 1989 - Tutorial on HMMs
  - Hamilton, 1989 - Markov-Switching models
  - Dempster et al., 1977 - EM Algorithm

## Guía de Troubleshooting

### Problemas Comunes

**1. Error de importación de modelos:**
```bash
# Verificar que estás en el entorno correcto
conda activate ritmo
python -c "from models import DLinear, PatchTST, TimeMixer, TimeXer"
```

**2. Error "No module named torch":**
```bash
# Reinstalar PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**3. Dataset no encontrado:**
```bash
# Verificar estructura de directorios
ls -la dataset/
# Descargar datasets según README.md original
```

**4. CUDA out of memory (si usas GPU):**
```bash
# Reducir batch size en script
--batch_size 16  # Default: 32
```

**5. Versiones incompatibles:**
```bash
# Recrear entorno desde environment.yml
conda env remove -n ritmo
conda env create -f environment.yml
```
---

**Recuerda**: Esta guía es la fuente única de verdad para el desarrollo del TFG RITMO. Mantenla actualizada a medida que el proyecto evoluciona. Cuando uses Claude Code, referencia esta guía para prácticas de desarrollo consistentes.
