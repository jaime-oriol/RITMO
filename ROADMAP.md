# ROADMAP - TFG RITMO

## Estado Actual del Repositorio

### Infraestructura Base (100% completa)

**Modelos Baseline (4):**
- DLinear.py - Descomposición + capas lineales
- PatchTST.py - Patch-based Transformer
- TimeMixer.py - Descomposición multi-escala
- TimeXer.py - Transformer con variables exógenas

**Componentes Reutilizables:**
- layers/StandardNorm.py - RevIN oficial (Kim et al., 2022)
- layers/Transformer_EncDec.py - Encoder/Decoder completo
- layers/SelfAttention_Family.py - Mecanismos de atención
- layers/Embed.py - PositionalEmbedding, TokenEmbedding
- exp/exp_long_term_forecasting.py - Framework training/test
- utils/metrics.py - MSE, MAE, RMSE, MAPE, MSPE
- utils/tools.py - EarlyStopping, adjust_learning_rate, visual
- data_provider/data_factory.py - Factory pattern datasets
- data_provider/data_loader.py - Loaders ETT, Weather, Electricity, etc.

**Datasets:**
- Entrenamiento HMM: ETTh1, ETTh2, Weather, Electricity
- Evaluación zero-shot: Traffic, Exchange

### Progreso Implementación (60% completado)

**✓ Módulo HMM (100% implementado):**
- ✓ Algoritmo Forward-Backward (hmm/forward_backward.py)
- ✓ Algoritmo Baum-Welch EM (hmm/baum_welch.py)
- ✓ Algoritmo de Viterbi (hmm/viterbi.py)
- ✓ Emisiones gaussianas (hmm/gaussian_emissions.py)
- ✓ Inicialización k-means (hmm/utils.py)
- ✗ Sistema guardado/carga parámetros (pendiente)

**✓ Módulo RevIN (100% implementado):**
- ✓ RevINNormalizer con gestión train/val/test (utils/revin.py)
- ✓ Validación reconstrucción norm→denorm
- ✓ Test con ETTh1 real validado

**→ Módulo Embeddings (0% implementado - SIGUIENTE):**
- Generación embeddings estructurados e_k = [μ_k, σ_k, A[k,:]]
- Mapeo tokens a embeddings (lookup table)
- Projection layer: Linear(2+K, d_model)
- Integración con PositionalEmbedding
- Gestión parámetros frozen (HMM) vs learnable (projection)

**Modelo RITMO (0% implementado):**
- Integración pipeline completo
- DataLoader personalizado
- Clase modelo final

---

## Descripción del TFG

**Título:** RITMO - Regímenes latentes mediante Inferencia Temporal con Markov Oculto para tokenización y forecasting de series temporales

**Objetivo:** Desarrollar sistema de tokenización basado en Hidden Markov Models donde estados ocultos actúen como embeddings latentes estructurados, evaluar desempeño en predicción long-term frente a técnicas determinísticas, y analizar trade-offs entre complejidad computacional, ratio de compresión e interpretabilidad.

**Pregunta de Investigación:** ¿Pueden los estados ocultos de un HMM actuar como embeddings latentes estructurados que capturen dependencias temporales y regímenes estadísticos de manera más efectiva que técnicas determinísticas actuales de tokenización para predicción de series temporales univariadas?

**Pipeline Propuesto (5 etapas):**

1. Normalización RevIN: X_norm = (X - μ) / σ
2. Entrenamiento HMM (Baum-Welch): estimar λ* = (A*, B*, π*)
3. Tokenización (Viterbi): Q* = argmax_Q P(Q|O, λ*)
4. Generación embeddings: e_k = [μ_k, σ_k, A[k,:]]
5. Predicción Transformer: ŷ_norm = Transformer([e_{z₁}, ..., e_{z_I}])

**IMPORTANTE: Dos Conceptos de "Token" en RITMO**

El término "token" tiene dos significados diferentes en este proyecto:

1. **Token para LLM** (Viterbi output):
   - Cada timestep → 1 token (estado k ∈ {0..K-1})
   - Serie continua [14.5, 14.7, 15.2, ...] → Secuencia discreta [2, 2, 1, ...]
   - Total: T tokens (mismo tamaño que serie original)
   - **Propósito:** Convertir valores continuos en símbolos discretos para procesamiento LLM

2. **Segmentos para compresión** (cambios de régimen):
   - Número de transiciones entre estados diferentes
   - Ejemplo: [2, 2, 2, 1, 1, 3, ...] → 3 segmentos (2 transiciones)
   - Ratio compresión = T / n_segmentos
   - **Propósito:** Medir eficiencia de captura de patrones temporales
   - **Según Anteproyecto línea 354:** "Para HMM corresponde al número de cambios de estado detectados por Viterbi"

**Nomenclatura en código:**
- `n_tokens_llm = len(states_pred)` → Tokens para LLM (T timesteps)
- `n_segments = np.sum(np.diff(states_pred) != 0) + 1` → Segmentos/regímenes
- `compression_ratio = T / n_segments` → Ratio de compresión

**Configuración Experimental:**
- Input length: I = 96 timesteps
- Prediction horizons: O ∈ {96, 192, 336, 720}
- Métricas: MSE, MAE
- Baselines: DLinear, PatchTST, TimeMixer, TimeXer

---

## Pasos a Seguir

### Fase 1: Implementar Módulo HMM

**1.1 Crear estructura de directorios:**
```
hmm/
├── __init__.py
├── forward_backward.py
├── baum_welch.py
├── viterbi.py
├── gaussian_emissions.py
└── utils.py
```

**1.2 Implementar algoritmo Forward-Backward:**
- Función forward: calcular α_t(k) = P(o₁...o_t, q_t=k|λ)
- Función backward: calcular β_t(k) = P(o_{t+1}...o_T|q_t=k, λ)
- Función gamma: calcular γ_t(k) = P(q_t=k|O, λ)
- Función xi: calcular ξ_t(k,l) = P(q_t=k, q_{t+1}=l|O, λ)
- Log-space arithmetic para estabilidad numérica

**1.3 Implementar algoritmo Baum-Welch:**
- Inicialización k-means sobre observaciones normalizadas
- Paso E: calcular γ_t(k) y ξ_t(k,l) usando forward-backward
- Paso M: re-estimar parámetros π, A, μ, σ²
- Criterio convergencia: |log P(O|λ^n) - log P(O|λ^{n-1})| < ε
- Iteración hasta convergencia

**1.4 Implementar algoritmo de Viterbi:**
- Variables δ_t(k): máxima probabilidad hasta t terminando en k
- Recursión: δ_t(k) = max_j [δ_{t-1}(j) × A_jk × b_k(o_t)]
- Backtracking: ψ_t(k) = argmax_j [δ_{t-1}(j) × A_jk]
- Output: secuencia óptima Q* = [z₁, ..., z_T]

**1.5 Implementar emisiones gaussianas:**
- Función b_k(o_t) = N(o_t; μ_k, σ²_k)
- Log-likelihood: log b_k(o_t) = -0.5×log(2πσ²_k) - (o_t - μ_k)²/(2σ²_k)
- Prevención underflow/overflow

**1.6 Sistema de guardado/carga:**
- Guardar λ* = (A*, B*, π*) en formato PyTorch
- Guardar parámetros RevIN (μ, σ) por dataset
- Cargar parámetros para inferencia

**1.7 Testing con datos sintéticos:**
- Generar series temporales desde HMM conocido
- Verificar recuperación de parámetros originales
- Validar convergencia Baum-Welch
- Validar precisión Viterbi

### Fase 2: Implementar Módulo Embeddings Estructurados

**2.1 Estructura de directorios:**
```
embeddings/
├── __init__.py
└── embedding_generator.py
```

**2.2 Implementar EmbeddingGenerator:**

**Input shape:**
- `tokens`: (batch_size, seq_len) - secuencia de estados k ∈ {0..K-1}
- `hmm_params`: dict con A, pi, mu, sigma (numpy arrays)

**Output shape:**
- `embeddings`: (batch_size, seq_len, 2+K) - embeddings estructurados

**Proceso:**
1. Crear lookup table: `embedding_table[k] = np.concatenate([mu[k], sigma[k], A[k,:]])`
   - Dimensión por estado: (2 + K)
   - mu[k]: float - centro del régimen k
   - sigma[k]: float - volatilidad del régimen k
   - A[k,:]: array[K] - probabilidades de transición desde k

2. Indexar: `embeddings = embedding_table[tokens]`
   - Operación vectorizada para batch completo

3. Convertir a torch.Tensor (device-aware)

**Parámetros frozen:**
- `hmm_params` (A, mu, sigma): **FROZEN** (no gradientes)
- Rationale: HMM captura estructura temporal, NO debe cambiar durante training Transformer

**2.3 Integración con Transformer:**

**Pipeline completo:**
```
1. EmbeddingGenerator: (batch, seq_len) → (batch, seq_len, 2+K)
2. Projection Layer: (batch, seq_len, 2+K) → (batch, seq_len, d_model)
3. PositionalEmbedding: (batch, seq_len, d_model) + pos_enc
4. Dropout(p=0.1)
5. Output listo para Transformer Encoder
```

**Parámetros learnable:**
- `projection_layer.weight`: Linear(2+K, d_model) - **LEARNABLE**
- `projection_layer.bias`: bias - **LEARNABLE**
- Rationale: Proyección aprende representación óptima para predicción

**Consideraciones de diseño:**
- d_model debe ser ≥ 2+K para evitar pérdida de información
- Recomendación: d_model = max(128, 2+K)
- No se concatena pos_enc antes de projection, se suma después
- Cache embeddings si dataset cabe en memoria (ETTh1: ~8K timesteps OK)

### Fase 3: Crear Modelo RITMO

**3.1 Implementar clase RITMO:**
```
models/RITMO.py
```
- Integrar RevIN (reutilizar layers/StandardNorm.py)
- Integrar generación embeddings
- Integrar Transformer Encoder (reutilizar layers/Transformer_EncDec.py)
- Projection head: d_model → pred_len
- Forward pass completo: raw data → normalización → embeddings → transformer → denormalización

**3.2 Registrar modelo:**
- Añadir 'RITMO': RITMO a exp/exp_basic.py:9-14
- Permite usar framework existente sin cambios

**3.3 DataLoader personalizado:**
```
data_provider/data_loader_ritmo.py
```
- Cargar datos raw
- Aplicar RevIN normalización
- Aplicar Viterbi tokenización
- Generar embeddings
- Output: embeddings pre-procesados
- Caching para eficiencia

### Fase 4: Entrenamiento HMM

**4.1 Script de entrenamiento HMM:**
- Cargar datasets: ETTh1, ETTh2, Weather, Electricity
- Aplicar RevIN a cada dataset
- Entrenar único HMM con K estados (determinar K vía AIC/BIC)
- Guardar λ* = (A*, B*, π*) entrenado
- Guardar parámetros RevIN (μ, σ) por dataset

**4.2 Análisis de estados ocultos:**
- Visualizar distribución temporal de activaciones
- Interpretar parámetros (μ_k, σ_k) de cada estado
- Analizar matriz de transición A
- Validar interpretabilidad de regímenes (día/noche, estaciones, etc.)

**4.3 Selección de K:**
- Probar K ∈ {3, 5, 7, 10}
- Calcular AIC = -2×log P(O|λ*) + 2×num_params
- Calcular BIC = -2×log P(O|λ*) + log(T)×num_params
- Seleccionar K óptimo

### Fase 5: Entrenamiento y Evaluación RITMO

**5.1 Configuración Transformer RITMO:**

**Arquitectura base (inspirada en PatchTST):**
- `d_model`: max(128, 2+K) - Asegurar ≥ dimensión embeddings
- `n_layers`: 3 - Encoder layers
- `n_heads`: 8 - Attention heads
- `d_ff`: 256 - Feedforward dimension (2×d_model típicamente)
- `dropout`: 0.1 - Regularización
- `activation`: 'gelu' - Función activación

**Projection head para predicción:**
- Input: (batch, seq_len, d_model) - Output encoder
- Pooling: mean over seq_len → (batch, d_model)
  - Alternativa: usar solo último timestep (batch, d_model)
- MLP: Linear(d_model, 128) → ReLU → Dropout(0.1) → Linear(128, pred_len)
- Output: (batch, pred_len) - Predicción horizonte O

**Hiperparámetros training:**
- `batch_size`: 32
- `learning_rate`: 1e-4 (AdamW optimizer)
- `warmup_epochs`: 10% del total (típicamente 3-5 epochs)
- `early_stopping_patience`: 10 epochs (monitor val_loss)
- `max_epochs`: 100
- `weight_decay`: 1e-4 (regularización L2)

**5.2 Entrenamiento modelo RITMO:**
- Reutilizar exp/exp_long_term_forecasting.py
- Modificar solo DataLoader (usar embeddings pre-generados)
- Early stopping automático con validación
- Learning rate schedule: ReduceLROnPlateau(factor=0.5, patience=5)
- Checkpoint best model según val_loss

**5.3 Evaluación en datasets benchmark:**
- ETTh1, ETTh2, Weather, Electricity (entrenamiento HMM)
- Horizontes: O ∈ {96, 192, 336, 720}
- Métricas: MSE, MAE
- Comparación con baselines (ya entrenados)
- Reportar promedio sobre 4 horizontes (formato TSLib)

**5.4 Protocolo evaluación zero-shot:**

**Datasets zero-shot (Traffic, Exchange):**

1. Cargar HMM params λ* entrenado en {ETTh1, ETTh2, Weather, Electricity}
2. **NO** re-entrenar HMM (params frozen)
3. Aplicar RevIN específico de Traffic/Exchange (fit en train split)
   - Cada dataset tiene su propio μ, σ para RevIN
4. Viterbi tokenización con λ* frozen
5. Generar embeddings con λ* frozen
6. Entrenar **solo** Transformer (projection + encoder + head)
   - HMM params NO tienen gradientes
7. Evaluar MSE/MAE en test split

**Validaciones zero-shot:**
- Verificar `log_likelihood(Traffic|λ*)` es razonable (no NaN/Inf)
- Comparar compresión ratio con datasets in-domain
- Analizar si estados activos son consistentes
- Reportar performance degradation vs. in-domain

**5.4 Análisis de trade-offs:**
- Ratio compresión: timesteps originales / número tokens (cambios de estado)
- Tiempo entrenamiento HMM (Baum-Welch)
- Tiempo inferencia (Viterbi)
- Throughput (muestras/segundo)
- MSE/MAE vs. complejidad computacional

**5.5 Interpretabilidad de regímenes:**
- Distribución temporal activaciones (periodicidad 24h en Electricity)
- Interpretación μ_k/σ_k (regímenes temperatura/consumo/volatilidad)
- Estructura transiciones A (estados persistentes vs. transitorios)
- Ejemplos: día/noche Electricity, estacionalidad Weather, bull/bear Exchange

### Fase 6: Documentación y Resultados

**6.1 Actualizar documentación:**
- Actualizar CLAUDE.md con pipeline implementado
- Documentar uso de módulo HMM
- Documentar reproducción de experimentos

**6.2 Tablas comparativas:**
- MSE/MAE promedio por dataset
- MSE/MAE por horizonte de predicción
- Ratio compresión HMM vs. PatchTST
- Tiempo entrenamiento/inferencia

**6.3 Visualizaciones:**
- Curvas predicción vs. ground truth
- Activación estados ocultos temporal
- Matriz de transición A como heatmap
- Distribución parámetros (μ_k, σ_k)

---

## Dependencias Adicionales Requeridas

**Bibliotecas Python:**
- scikit-learn (ya disponible) - k-means para inicialización HMM
- scipy (ya disponible) - funciones estadísticas

**Opcional:**
- hmmlearn - validar implementación custom HMM

---

## Referencias Clave

**Fundamentos HMM:**
- Rabiner (1989) - Tutorial on HMMs and selected applications
- Dempster et al. (1977) - EM algorithm

**RevIN:**
- Kim et al. (2022) - Reversible Instance Normalization (ICLR)

**Baselines:**
- DLinear: Zeng et al. (2023) - AAAI
- PatchTST: Nie et al. (2023) - ICLR
- TimeMixer: Wang et al. (2024) - ICLR
- TimeXer: Wang et al. (2024) - NeurIPS

**Datasets:**
- Time-Series-Library (TSLib): https://github.com/thuml/Time-Series-Library

---

## Estimación de Código

**Código nuevo requerido:**
- Módulo HMM: ~730 líneas
- Módulo embeddings: ~210 líneas
- Modelo RITMO: ~200 líneas
- DataLoader custom: ~150 líneas
- Total: ~1,290 líneas

**Código reutilizable:**
- Normalización RevIN: ~70 líneas
- Transformer completo: ~500 líneas
- Framework experimentación: ~330 líneas
- Total: ~900 líneas

**Ratio código nuevo/reutilizable:** 60/40
