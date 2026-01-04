# Técnicas de Tokenización para Series Temporales

## Resumen

Comparación de 6 técnicas de tokenización aplicadas a forecasting de series temporales. Cada técnica convierte la serie cruda en tokens procesables por modelos de aprendizaje automático.

## 1. HMM (Viterbi) - RITMO

**Paper:** Tesis RITMO (2025)
**Tipo:** Probabilístico

### Qué hace
Entrena un Hidden Markov Model con K estados ocultos. Cada estado representa un "régimen" estadístico de la serie. Usa algoritmo Viterbi para encontrar la secuencia óptima de estados.

### Cómo funciona
1. Entrena HMM con Baum-Welch (EM algorithm)
2. Aprende parámetros: A (transiciones), π (inicial), μ (medias), σ (volatilidades)
3. Viterbi encuentra Q* = argmax P(Q|O,λ)
4. Cada timestep se asigna a un estado (token)

### Vocabulario y Compresión
- Vocabulario: K=5 tokens (estados)
- Secuencia: T tokens (1 por timestep)
- Compresión: Run-length encoding de cambios de estado
- Ratio: 27x en ETTh1 (8640 timesteps → 318 segmentos)

### Ventajas
- Tokens tienen significado estadístico (μ, σ por token)
- Captura dinámicas temporales (matriz A)
- Embeddings estructurados e_k = [μ_k, σ_k, A[k,:]]
- Interpretable

### Desventajas
- Requiere entrenamiento HMM (costo computacional)
- Asume distribuciones gaussianas

## 2. SAX (Symbolic Aggregate approXimation)

**Paper:** Lin et al. (2007)
**Tipo:** Discretización

### Qué hace
Divide el espacio de valores en N regiones equiprobables bajo distribución gaussiana. Asigna un símbolo alfabético a cada valor según su región.

### Cómo funciona
1. Normaliza serie: z = (x - μ) / σ
2. Calcula breakpoints gaussianos: Φ^(-1)(k/N) para k=1..N-1
3. Asigna símbolo según breakpoint: np.searchsorted(breakpoints, z)
4. Resultado: secuencia de símbolos {'a', 'b', 'c', ...}

### Vocabulario y Compresión
- Vocabulario: 8 símbolos (alphabet_size=8 típico)
- Secuencia: T tokens (1 por timestep)
- Compresión: 1.0x (sin compresión, mapeo 1:1)

### Ventajas
- Muy simple de implementar
- Reduce valores continuos a alfabeto finito
- Interpretable (cada símbolo = rango de valores)

### Desventajas
- Sin compresión (1 token por timestep)
- Pierde información (discretización irreversible)
- Asume distribución gaussiana

## 3. LLMTime (Text-based)

**Paper:** Gruver et al. (2023)
**Tipo:** Textual

### Qué hace
Convierte cada valor numérico a representación textual en base N. Separa dígitos con espacios para que LLMs puedan procesarlos secuencialmente.

### Cómo funciona
1. Redondea valor a precisión decimal (ej: 2 decimales)
2. Separa en signo + dígitos enteros + punto + dígitos decimales
3. Convierte cada dígito a caracter separado por espacios
4. Ejemplo: 23.45 → " 2 3 . 4 5"

### Vocabulario y Compresión
- Vocabulario: 13 caracteres (base=10: dígitos 0-9, signo, punto, separador)
- Secuencia: ~10x más tokens que timesteps (expansión)
- Compresión: 0.10x (expansión, no compresión)

### Ventajas
- Compatible con LLMs pre-entrenados (GPT, etc.)
- No requiere arquitectura especializada
- Puede aprovechar conocimiento lingüístico del LLM

### Desventajas
- Expansión de datos (10x más tokens)
- Longitud variable según magnitud del valor
- Computacionalmente costoso (procesar caracteres)

## 4. PatchTST (Patch-based)

**Paper:** Nie et al. (2023)
**Tipo:** Segmentación

### Qué hace
Divide serie en patches (ventanas) de longitud fija. Cada patch es un token multidimensional.

### Cómo funciona
1. Desliza ventana de tamaño patch_len con paso stride
2. Usa torch.unfold() para generar patches
3. Cada patch ∈ R^patch_len es un token
4. Ejemplo: patch_len=16, stride=16 → patches non-overlapping

### Vocabulario y Compresión
- Vocabulario: Continuo (cada patch es vector R^16)
- Secuencia: T/patch_len patches
- Compresión: 16x (para patch_len=16)

### Ventajas
- Reduce complejidad Transformer de O(T²) a O((T/P)²)
- Captura estructura local (ventana de contexto)
- Sin entrenamiento previo necesario

### Desventajas
- Patches arbitrarios (no tienen significado estadístico)
- Trunca timesteps finales si T no es múltiplo de patch_len
- Pierde información inter-patch si non-overlapping

## 5. Autoformer (Decomposition)

**Paper:** Wu et al. (2021)
**Tipo:** Descomposición

### Qué hace
Descompone serie en 2 componentes: Trend (baja frecuencia) y Seasonal (alta frecuencia). Cada componente es un "token" que se modela independientemente.

### Cómo funciona
1. Calcula trend mediante promedio móvil: AvgPool1d(kernel_size=25)
2. Seasonal = serie - trend (residuo)
3. Modela cada componente por separado
4. Reconstrucción: serie = trend + seasonal

### Vocabulario y Compresión
- Vocabulario: 2 componentes (trend, seasonal)
- Secuencia: 2 tokens (cada uno de longitud T)
- Compresión: 750x (T/2 para T=1500)

### Ventajas
- Separación frecuencias (trend suave, seasonal oscilante)
- Reconstrucción exacta (trend + seasonal = serie)
- Muy simple de implementar

### Desventajas
- Solo 2 tokens (representación muy comprimida)
- Kernel_size fijo (hiperparámetro sensible)
- No captura regímenes complejos

## 6. MOMENT (Foundation Model)

**Paper:** Goswami et al. (2024)
**Tipo:** Pre-training con masking

### Qué hace
Combina patching con masking aleatorio. Modelo aprende a reconstruir patches enmascarados durante pre-training (self-supervised).

### Cómo funciona
1. Segmenta en patches (como PatchTST)
2. Enmascara 30% de patches aleatoriamente
3. Modelo predice patches enmascarados
4. Embeddings aprendidos se usan para downstream tasks

### Vocabulario y Compresión
- Vocabulario: Continuo (embeddings learnables)
- Secuencia: T/patch_len patches (30% masked)
- Compresión: 16x (para patch_len=16)

### Ventajas
- Pre-training self-supervised (no requiere labels)
- Aprende representaciones generales
- Transfer learning a múltiples datasets

### Desventajas
- Requiere fase de pre-training (costosa)
- Masking introduce ruido durante entrenamiento
- Depende de calidad del pre-training

## Tabla Comparativa

| Técnica | Vocabulario | Tokens (T=1500) | Segmentos | CR | Tipo |
|---------|------------|-----------------|-----------|----|----|
| HMM | K=5 | 1500 | 25-318 | 27x | Probabilístico |
| SAX | 8 símbolos | 1500 | 1500 | 1x | Discreto |
| LLMTime | 13 chars | 15000 | N/A | 0.1x | Textual |
| PatchTST | Continuo | 93 patches | 93 | 16x | Segmentación |
| Autoformer | 2 componentes | 2 | 2 | 750x | Descomposición |
| MOMENT | Continuo (masked) | 93 patches | 93 | 16x | Pre-training |

## Métricas Clave

**Compression Ratio (CR):**
- Mayor CR = mejor compresión
- CR > 1: compresión (HMM, PatchTST, Autoformer, MOMENT)
- CR = 1: sin compresión (SAX)
- CR < 1: expansión (LLMTime)

**Vocabulario:**
- Discreto: SAX (8 símbolos), HMM (K estados)
- Continuo: PatchTST, MOMENT (embeddings)
- Textual: LLMTime (caracteres)
- Componentes: Autoformer (trend + seasonal)

**Interpretabilidad:**
- Alta: HMM (estados estadísticos), SAX (símbolos), Autoformer (componentes)
- Media: PatchTST (ventanas locales)
- Baja: MOMENT (embeddings learnables black box), LLMTime (caracteres)

## Implementación

Todas las técnicas están implementadas en `tecnicas/`:
- `discretization.py`: SAX
- `text_based.py`: LLMTime
- `patching.py`: PatchTST
- `decomposition.py`: Autoformer
- `foundation.py`: MOMENT
- HMM: `hmm/` (baum_welch.py, viterbi.py)

Notebooks de comparación:
- `tecnicas/ETTh2_tokenization.ipynb`
- `tecnicas/Electricity_tokenization.ipynb`

## Conclusión

Cada técnica tiene trade-offs:

**Para forecasting determinístico:** PatchTST (buena compresión, simple)
**Para interpretabilidad:** HMM (embeddings estructurados) o SAX (símbolos discretos)
**Para LLMs pre-entrenados:** LLMTime (compatible con GPT)
**Para descomposición frecuencias:** Autoformer (trend-seasonal)
**Para transfer learning:** MOMENT (foundation model)

**RITMO** combina ventajas de HMM (interpretabilidad + embeddings estructurados) con eficiencia computacional (compresión 27x).
