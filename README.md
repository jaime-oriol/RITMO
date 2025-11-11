# RITMO: Tokenización de Series Temporales mediante Hidden Markov Models

**Regímenes latentes mediante Inferencia Temporal con Markov Oculto para tokenización y forecasting de series temporales**

## Descripción del Proyecto

Este repositorio contiene la implementación del Trabajo de Fin de Grado titulado "Tokenizar series temporales con técnicas existentes y contrastar resultados empleando estados ocultos de Markov". El proyecto investiga la aplicación de Hidden Markov Models (HMM) como mecanismo de tokenización probabilística para series temporales univariadas en el contexto de modelos transformer.

## Pregunta de Investigación

¿Pueden los estados ocultos de un Hidden Markov Model actuar como embeddings latentes estructurados que capturen dependencias temporales y regímenes estadísticos de manera más efectiva que las técnicas determinísticas actuales de tokenización para la predicción de series temporales univariadas?

## Motivación

La aplicación de arquitecturas transformer a series temporales enfrenta un desafío fundamental: las series temporales son inherentemente continuas, mientras que los transformers operan sobre secuencias discretas de tokens. Esta disparidad requiere una transformación previa denominada tokenización, que constituye un cuello de botella crítico en el pipeline completo.

Los métodos existentes (discretización, patching, descomposición, foundation models) presentan limitaciones inherentes:
- Discretización: granularidad fija sin adaptación a variabilidad intrínseca
- Patching: segmentación arbitraria que puede fragmentar patrones coherentes
- Foundation models: embeddings implícitos sin significado estadístico explícito

En contraposición, los Hidden Markov Models ofrecen estructura probabilística que modela explícitamente transiciones entre estados ocultos mediante cadenas de Markov de primer orden, proporcionando representaciones más ricas que métodos determinísticos actuales.

## Metodología

### Pipeline RITMO Propuesto

<p align="center">
<img src="./pic/Pipeline-RITMO.png" alt="Pipeline RITMO" width="800"/>
</p>

El sistema implementa un pipeline de cinco etapas:

**1. Normalización (RevIN)**
```
X_norm = (X - μ) / σ
```
Reversible Instance Normalization para garantizar estacionariedad y robustez frente a distribution shift.

**2. Entrenamiento HMM (Baum-Welch)**
- Estimación de parámetros λ* = (A*, B*, π*) mediante algoritmo EM
- Emisiones gaussianas N(μ_k, σ²_k) por estado
- Entrenamiento sobre datasets: ETTh1, ETTh2, Weather, Electricity
- Complejidad: O(T·K²) por iteración

**3. Tokenización (Viterbi)**
```
Q* = argmax_Q P(Q|O, λ*)
```
Programación dinámica para obtener secuencia óptima de estados ocultos [z₁, z₂, ..., z_T].

**4. Generación de Embeddings Estructurados**
```
e_k = [μ_k, σ_k, A[k,:]]
```
Donde:
- μ_k: centro del régimen estadístico
- σ_k: volatilidad intrínseca del régimen
- A[k,:]: probabilidades de transición (dinámica temporal explícita)

**5. Predicción (Transformer)**
```
ŷ_norm = Transformer([e_{z₁}, ..., e_{z_I}])
ŷ = ŷ_norm × σ + μ
```
Transformer decoder opera sobre embeddings estructurados en lugar de valores crudos.

## Modelos Baseline

El proyecto compara el enfoque HMM propuesto contra cuatro baselines representativos del estado del arte:

### 1. PatchTST (Nie et al., 2023)
Baseline principal con patch-based tokenization. Segmenta series en patches como tokens, reduciendo complejidad de atención de O(L²) a O((L/S)²). Implementa channel-independence y normalización Non-stationary.

**Archivo:** `models/PatchTST.py`

### 2. DLinear (Zeng et al., 2023)
Baseline simple que demuestra que descomposición lineal puede superar a transformers complejos. Descompone series en trend + seasonal aplicando capas lineales separadas.

**Archivo:** `models/DLinear.py`

### 3. TimeMixer (Wang et al., 2024)
Baseline con descomposición multi-escala y mixing operations. Implementa Past Decomposable Mixing (PDM) y soporta moving average y DFT decomposition.

**Archivo:** `models/TimeMixer.py`

### 4. TimeXer (Wang et al., 2024)
Baseline para forecasting con variables exógenas. Implementa paradigma de forecasting práctico con embeddings endógenos/exógenos y global token mechanism.

**Archivo:** `models/TimeXer.py`

## Datasets

### Entrenamiento HMM (4 datasets)

1. **ETTh1** - Electric Transformer Temperature (hourly)
   - 7 características, frecuencia horaria
   - Split 7:1:2 (train/val/test)
   - Benchmark universal consolidado

2. **ETTh2** - Electric Transformer Temperature (hourly)
   - Variante con distribution shift documentado
   - Valida robustez de RevIN ante cambios distribucionales

3. **Weather** - Temperatura wet-bulb
   - Regímenes climáticos estacionales claros (invierno/verano)
   - Ideal para evaluar captura de cambios de régimen interpretables

4. **Electricity** - Consumo eléctrico (MT_320)
   - Periodicidad diaria fuerte (24h)
   - Validación de patrones cíclicos día/noche

### Evaluación Zero-Shot (2 datasets)

5. **Traffic** - Ocupación de sensores de tráfico
   - Dominio diferente sin re-entrenamiento
   - Regímenes rush-hour explícitos

6. **Exchange** - Tipos de cambio
   - Series financieras sin periodicidad marcada
   - Test de robustez para HMM en series sin estacionalidad clara

**Nota:** Descargar datasets desde [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) o [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) y colocar en `./dataset/`.

## Configuración Experimental

**Protocolo estándar:**
- Input length: I = 96 timesteps
- Prediction horizons: O ∈ {96, 192, 336, 720}
- Métricas: MSE (Mean Squared Error), MAE (Mean Absolute Error)
- Configuración: Look-Back-96 para comparabilidad directa
- Reporte: Promedio sobre los cuatro horizontes de predicción

**Protocolo de evaluación:**
1. Desempeño predictivo (MSE/MAE por dataset y promedio general)
2. Ratio de compresión (timesteps originales / tokens generados)
3. Eficiencia computacional (tiempo de entrenamiento/inferencia)
4. Interpretabilidad de regímenes (solo HMM: análisis cualitativo de estados)

## Instalación

### Requisitos

- Python 3.10
- PyTorch 2.9.0+ (CPU o GPU)
- Conda (recomendado para gestión de entorno)

### Setup del Entorno

```bash
# Opción 1: Crear entorno desde environment.yml (recomendado)
conda env create -f environment.yml
conda activate ritmo

# Opción 2: Instalación manual
conda create -n ritmo python=3.10 -y
conda activate ritmo
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install einops==0.8.0 local-attention==1.9.14 matplotlib pandas scikit-learn scipy sktime reformer-pytorch tqdm PyWavelets patool
```

### Verificación de Instalación

```bash
# Verificar importación de modelos
python -c "from models import DLinear, PatchTST, TimeMixer, TimeXer; print('Modelos importados correctamente')"

# Verificar PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## Uso

### Ejecutar Experimentos Baseline

```bash
# Ejecutar modelo específico en dataset específico
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
--seq_len: Longitud de entrada (default: 96)
--pred_len: Horizonte de predicción (96 | 192 | 336 | 720)
--features: S (univariate) | M (multivariate) | MS (multi-to-uni)
```

### Ejecutar Scripts Pre-configurados

Los scripts incluyen configuración óptima para múltiples horizontes de predicción:

```bash
# Long-term forecasting
bash scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh
bash scripts/long_term_forecast/Weather_script/TimeMixer.sh
bash scripts/long_term_forecast/ECL_script/DLinear.sh

# Exogenous forecasting (TimeXer)
bash scripts/exogenous_forecast/ETTh1/TimeXer.sh
```

## Estructura del Repositorio

```
RITMO/
├── models/                  # Modelos baseline (4 modelos)
│   ├── DLinear.py
│   ├── PatchTST.py
│   ├── TimeMixer.py
│   └── TimeXer.py
├── layers/                  # Componentes compartidos
│   ├── Embed.py
│   ├── Transformer_EncDec.py
│   ├── SelfAttention_Family.py
│   ├── Autoformer_EncDec.py
│   └── StandardNorm.py
├── exp/                     # Clases de experimentación
│   ├── exp_basic.py
│   └── exp_long_term_forecasting.py
├── data_provider/          # Carga de datos
│   ├── data_factory.py
│   └── data_loader.py
├── utils/                   # Utilidades
│   ├── metrics.py
│   ├── tools.py
│   └── timefeatures.py
├── scripts/                 # Scripts de ejecución (26 scripts)
│   ├── long_term_forecast/
│   └── exogenous_forecast/
├── run.py                   # Script principal
├── environment.yml          # Especificación Conda
├── Anteproyecto-RITMO.md   # Documento completo del TFG
└── CLAUDE.md                # Guía de desarrollo
```

## Marco Teórico

### Hidden Markov Models

Un HMM se define mediante el conjunto de parámetros λ = (A, B, π):

- **A**: Matriz de transición K×K con A_ij = P(q_t = j | q_{t-1} = i)
- **B**: Distribuciones de emisión gaussianas N(μ_k, σ²_k) por estado k
- **π**: Distribución inicial π_k = P(q_1 = k)

**Algoritmos fundamentales:**

1. **Baum-Welch (entrenamiento):** Estimación de máxima verosimilitud mediante EM. Garantiza convergencia monótona a máximo local.

2. **Viterbi (decodificación):** Encuentra secuencia óptima Q* = argmax_Q P(Q|O, λ) mediante programación dinámica con complejidad O(T·K²).

3. **Forward-Backward (inferencia):** Calcula probabilidades marginales P(q_t = k | O, λ) para cuantificar incertidumbre en asignación de estados.

### Embeddings Estructurados vs. Determinísticos

La propuesta HMM genera embeddings donde cada dimensión posee significado estadístico interpretable:

```
e_k = [μ_k, σ_k, A[k,:]]
```

A diferencia de métodos determinísticos (discretización, patching) o implícitos (foundation models), esta representación encapsula simultáneamente:
- Información estadística local (μ_k caracteriza valor central, σ_k cuantifica volatilidad)
- Dinámica temporal explícita (A[k,:] codifica P(z_t = j | z_{t-1} = k) directamente)

## Estado del Arte

### Tokenización para Series Temporales

**Discretización:**
- SAX (Lin et al., 2007): Symbolic Aggregate approXimation
- VQ-VAE (van den Oord et al., 2017): Vector Quantized Variational AutoEncoders
- Chronos (Ansari et al., 2024): Cuantización uniforme + T5/GPT-2

**Patching:**
- PatchTST (Nie et al., 2023): Patches de longitud fija como tokens
- EntroPE (Abeywickrama et al., 2025): Boundaries adaptativos mediante entropía condicional

**Descomposición:**
- Autoformer (Wu et al., 2021): Descomposición con autocorrelación
- FEDformer (Zhou et al., 2022): Descomposición en dominio frecuencial
- TimeMixer (Wang et al., 2024): Descomposición multi-escala

**Foundation Models:**
- MOMENT (Goswami et al., 2024): Pre-entrenado sobre 27B+ timesteps
- Timer (Liu et al., 2024): GPT-2 con discretización cuantílica adaptativa

### HMM en Series Temporales

**Fundamentos:**
- Rabiner (1989): Tutorial sistemático de HMM
- Dempster et al. (1977): Algoritmo EM para estimación de parámetros
- Hamilton (1989): Markov-Switching para series económicas

**Aplicaciones Modernas:**
- Tang & Matteson (2021): ProTran combina State-Space Models con attention
- Yeh & Tang (2022): Neural HMMs con dependencias markovianas explícitas
- Fox et al. (2011): Sticky HDP-HMM con selección automática de estados

### Saturación de Benchmarks

Wang et al. (2025) establecen "Accuracy Law" que caracteriza relación exponencial entre complejidad de patrones y error mínimo: MSE ≈ exp(α·Complexity). Identifican saturación en múltiples benchmarks (ETTh1, ETTh2, Weather, Electricity), donde métodos determinísticos actuales se aproximan asintóticamente a límite teórico.

Esta saturación posiciona la integración de HMM como dirección necesaria para avanzar más allá de enfoques determinísticos mediante estructura probabilística explícita.

## Referencias Principales

### Modelos Baseline

Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. *Proceedings of the 11th International Conference on Learning Representations*.

Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are transformers effective for time series forecasting? *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(9), 11121-11128.

Wang, Y., Wu, H., Dong, J., Liu, Y., Long, M., & Wang, J. (2024). TimeMixer: Decomposable multiscale mixing for time series forecasting. *Proceedings of the 12th International Conference on Learning Representations*.

Wang, Y., Wu, H., Dong, J., Liu, Y., Qiu, Y., Zhang, H., Wang, J., & Long, M. (2024). TimeXer: Empowering transformers for time series forecasting with exogenous variables. *Advances in Neural Information Processing Systems 37*.

### Hidden Markov Models

Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.

Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Methodological)*, 39(1), 1-38.

### Surveys

Wen, Q., Zhou, T., Zhang, C., Chen, W., Ma, Z., Yan, J., & Sun, L. (2023). Transformers in time series: A survey. *Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence*.

Zhang, X., Chowdhury, R. R., Gupta, R. K., & Shang, J. (2024). Large language models for time series: A survey. *arXiv preprint arXiv:2402.01801*.

Liang, Y., Wen, H., Nie, Y., Jiang, Y., Jin, M., Song, D., Pan, S., & Wen, Q. (2024). Foundation models for time series analysis: A tutorial and survey. *Proceedings of the 30th ACM SIGKKDD Conference on Knowledge Discovery and Data Mining*.

### Benchmarks

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(12), 11106-11115.

Wang, Y., Wu, H., Ma, Y., Fang, Y., Zhang, Z., Liu, Y., Wang, S., Ye, Z., Xiang, Y., Wang, J., & Long, M. (2025). Accuracy law for the future of deep time series forecasting. *arXiv preprint arXiv:2510.02729*.

## Documentación Adicional

Para información detallada sobre la metodología, consultar:
- **Anteproyecto-RITMO.md**: Memoria justificativa completa con estado del arte, marco teórico y cronograma
- **CLAUDE.md**: Guía de desarrollo con estándares de código y workflow

## Licencia

Este proyecto se basa en Time-Series-Library (TSLib) y mantiene su licencia original. Todos los datasets utilizados son públicos y se obtienen de las fuentes originales listadas en la documentación.

## Contacto

Jaime Oriol Goicoechea
Universidad del País Vasco (UPV/EHU)
Trabajo de Fin de Grado - Ingeniería Informática

Directores:
[Información de contacto de los directores del TFG]

Para preguntas o sugerencias sobre el proyecto, abrir un issue en el repositorio o contactar directamente.

## Agradecimientos

Este proyecto se construye sobre la base de Time-Series-Library (TSLib) desarrollado por THUML (Tsinghua Machine Learning Lab). Agradecemos a los autores originales y mantenedores por proporcionar una base de código robusta y bien documentada.

Repositorio original: https://github.com/thuml/Time-Series-Library
