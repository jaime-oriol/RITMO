"""
Métricas intrínsecas para evaluar tokenización SIN tareas downstream.
Permiten comparar técnicas de tokenización usando solo los tokens generados.
No requieren entrenar modelos ni ejecutar forecasting/clasificación.

Referencias:
    - Chiarot & Silvestri 2022: Time Series Compression Survey (ACM)
    - Uzan et al. 2024: Greed is All You Need (ACL)
    - CAMEO 2025: ACF/PACF retention para dependencias temporales
"""

import numpy as np  # Operaciones matemáticas con arrays
from scipy import stats  # Para calcular entropía
from typing import Dict, Any  # Anotaciones de tipos


def compression_ratio(T_original: int, num_tokens: int) -> float:
    """
    Ratio de compresión: cuánto reduce la tokenización.
    Mide eficiencia de representación.

    Entrada:
        T_original: Longitud de la serie original (número de timesteps)
        num_tokens: Número de tokens generados por la técnica

    Salida:
        Ratio de compresión:
        - >1 significa que comprime (menos tokens que timesteps)
        - =1 significa sin compresión (un token por timestep)
        - <1 significa expansión (más tokens que timesteps)

    Ejemplo:
        Serie de 1000 puntos → 62 patches = ratio 16.1x
        Serie de 1000 puntos → 10496 caracteres (LLMTime) = ratio 0.095x

    Ref: Chiarot & Silvestri 2022 "Time Series Compression Survey"
    """
    # Evitar división por cero
    if num_tokens == 0:
        return float('inf')

    # Fórmula simple: timesteps originales / tokens generados
    return T_original / num_tokens


def mse_reconstruction(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Error cuadrático medio de reconstrucción.
    Mide cuánta información se pierde al tokenizar y reconstruir.

    Entrada:
        original: Serie temporal original [T]
        reconstructed: Serie reconstruida desde tokens [T]

    Salida:
        MSE: Error cuadrático medio
        - 0 = reconstrucción perfecta (sin pérdida)
        - Mayor valor = más pérdida de información

    Interpretación:
        MSE bajo = tokenización preserva bien la señal
        MSE alto = tokenización pierde información importante

    Ref: Chiarot & Silvestri 2022
    """
    # Ajustar longitudes si difieren (algunas técnicas pueden truncar)
    min_len = min(len(original), len(reconstructed))

    # Calcular diferencia al cuadrado, promediar
    error = (original[:min_len] - reconstructed[:min_len])**2
    return float(np.mean(error))


def acf_retention(original: np.ndarray, reconstructed: np.ndarray, nlags: int = 20) -> float:
    """
    Retención de autocorrelación: preservación de dependencias temporales.
    Mide si la tokenización mantiene la estructura temporal de la serie.

    Idea: Si la ACF de la reconstrucción es similar a la original,
    la tokenización preserva las dependencias entre timesteps.

    Entrada:
        original: Serie temporal original [T]
        reconstructed: Serie reconstruida desde tokens [T]
        nlags: Número de lags para calcular ACF (default: 20)

    Salida:
        Correlación de Pearson entre ACF original y ACF reconstruida:
        - 1.0 = preservación perfecta de dependencias
        - 0.0 = sin relación
        - -1.0 = dependencias invertidas

    Ejemplo:
        PatchTST con patches non-overlapping: ACF retention ~0.99
        SAX con discretización agresiva: ACF retention ~0.85

    Ref: CAMEO 2025
    """
    # Ajustar longitudes
    min_len = min(len(original), len(reconstructed))

    # ACF necesita suficientes datos (al menos 4x los lags)
    nlags = min(nlags, min_len // 4)

    def _acf(x, nlags):
        """
        Calcula autocorrelación manual.
        Evita dependencia de statsmodels para mantener código ligero.
        """
        n = len(x)
        x = x - np.mean(x)  # Centrar la serie (media = 0)

        # Autocorrelación = correlación de x consigo misma desplazada
        # np.correlate da correlación cruzada completa
        acf_vals = np.correlate(x, x, mode='full')[n-1:n+nlags]

        # Normalizar por varianza (acf[0] = 1.0)
        if acf_vals[0] == 0:
            return np.zeros(nlags + 1)  # Serie constante
        return acf_vals / acf_vals[0]

    # Calcular ACF de ambas series
    acf_orig = _acf(original[:min_len], nlags)
    acf_recon = _acf(reconstructed[:min_len], nlags)

    # Correlación de Pearson entre las dos ACFs
    return float(np.corrcoef(acf_orig, acf_recon)[0, 1])


def vocabulary_entropy(tokens: np.ndarray) -> float:
    """
    Entropía del vocabulario: distribución de uso de tokens.
    Mide qué tan uniformemente se usan los diferentes tokens.

    Idea: Si todos los tokens se usan igual, entropía máxima.
    Si un token domina, entropía baja.

    Entrada:
        tokens: Secuencia de tokens discretos [T]
                (ej: [0, 2, 1, 1, 3, 2, 0, ...] para SAX)

    Salida:
        Entropía normalizada en rango [0, 1]:
        - 1.0 = uso perfectamente uniforme del vocabulario
        - 0.0 = solo se usa un token

    Ejemplo:
        SAX con 8 símbolos usados equitativamente: ~0.99
        HMM con 1 estado dominante: ~0.60

    Ref: Uzan et al. 2024 "Greed is All You Need"
    """
    # Contar ocurrencias de cada token único
    _, counts = np.unique(tokens, return_counts=True)

    # Convertir conteos a probabilidades
    probs = counts / counts.sum()

    # Calcular entropía de Shannon (base 2 = bits)
    entropy = stats.entropy(probs, base=2)

    # Entropía máxima = log2(número de tokens únicos)
    max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1

    # Normalizar a [0, 1]
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def bigram_entropy(tokens: np.ndarray) -> float:
    """
    Entropía de bigramas: distribución de transiciones entre tokens consecutivos.
    Mide estructura secuencial y predictibilidad de transiciones.

    Idea: Si token_t predice perfectamente token_{t+1}, entropía baja.
    Si transiciones son uniformes, entropía alta.

    Entrada:
        tokens: Secuencia de tokens discretos [T]

    Salida:
        Entropía normalizada de bigramas en [0, 1]:
        - 1.0 = transiciones completamente uniformes
        - 0.0 = transiciones determinísticas

    Ejemplo:
        HMM con matriz A suave: ~0.85
        SAX con alta variabilidad: ~0.95

    Ref: Uzan et al. 2024 "Greed is All You Need"
    """
    if len(tokens) < 2:
        return 0.0

    # Crear bigramas (token_t, token_{t+1})
    bigrams = list(zip(tokens[:-1], tokens[1:]))

    # Contar ocurrencias de cada bigrama
    from collections import Counter
    bigram_counts = Counter(bigrams)

    # Convertir a probabilidades
    total = len(bigrams)
    probs = np.array([count / total for count in bigram_counts.values()])

    # Entropía de Shannon
    entropy = stats.entropy(probs, base=2)

    # Entropía máxima = log2(número de bigramas únicos observados)
    max_entropy = np.log2(len(bigram_counts)) if len(bigram_counts) > 1 else 1

    # Normalizar
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def token_persistence(tokens: np.ndarray) -> float:
    """
    Persistencia de tokens: longitud media de runs consecutivos.
    Mide cuánto tiempo permanece en el mismo token/estado.

    Idea: HMM con regímenes persistentes tendrá runs largos.
    SAX con alta variabilidad tendrá runs cortos.

    Entrada:
        tokens: Secuencia de tokens discretos [T]

    Salida:
        Longitud media de runs (≥1):
        - 1.0 = cada token es diferente del anterior
        - >1 = tokens consecutivos iguales (persistencia)

    Ejemplo:
        HMM: [0,0,0,1,1,2,2,2,2] → runs [3,2,4] → mean=3.0
        SAX: [0,1,2,3,4,5,6,7] → runs [1,1,...] → mean=1.0

    Ref: Run-length encoding (lossless compression)
    """
    if len(tokens) == 0:
        return 0.0

    if len(tokens) == 1:
        return 1.0

    # Detectar cambios de token
    changes = np.diff(tokens) != 0
    change_indices = np.where(changes)[0] + 1

    # Añadir inicio y fin para delimitar runs
    run_boundaries = np.concatenate([[0], change_indices, [len(tokens)]])

    # Calcular longitud de cada run
    run_lengths = np.diff(run_boundaries)

    # Longitud media
    return float(np.mean(run_lengths))


def top_k_coverage(tokens: np.ndarray, k: int = 5) -> float:
    """
    Cobertura top-K: fracción de uso cubierta por los K tokens más frecuentes.
    Mide concentración de la distribución de tokens.

    Idea: Si vocabulario está balanceado, top-K coverage es bajo.
    Si pocos tokens dominan, top-K coverage es alto.

    Entrada:
        tokens: Secuencia de tokens discretos [T]
        k: Número de tokens más frecuentes a considerar (default: 5)

    Salida:
        Fracción [0, 1] del uso total cubierto por top-K:
        - 1.0 = los top-K cubren todo el uso (vocabulario concentrado)
        - ~0.X = distribución más balanceada

    Ejemplo:
        HMM con 1 estado dominante: top-1 coverage ~0.70
        SAX balanceado: top-5 coverage ~0.62

    Ref: Zipf's law, Power laws in NLP
    """
    if len(tokens) == 0:
        return 0.0

    # Contar frecuencias
    unique, counts = np.unique(tokens, return_counts=True)

    # Ordenar por frecuencia descendente
    sorted_counts = np.sort(counts)[::-1]

    # Tomar top-K (o menos si vocab_size < k)
    k_actual = min(k, len(sorted_counts))
    top_k_counts = sorted_counts[:k_actual]

    # Fracción cubierta
    coverage = top_k_counts.sum() / counts.sum()

    return float(coverage)


def evaluate_tokenization(original: np.ndarray,
                          reconstructed: np.ndarray,
                          num_tokens: int,
                          tokens: np.ndarray = None) -> Dict[str, float]:
    """
    Evalúa tokenización con todas las métricas intrínsecas.
    Función de conveniencia que calcula todas las métricas a la vez.

    Entrada:
        original: Serie temporal original [T]
        reconstructed: Serie reconstruida desde tokens [T]
        num_tokens: Número de tokens generados
        tokens: Secuencia de tokens discretos (opcional, para métricas discretas)

    Salida:
        Diccionario con todas las métricas:
        - compression_ratio: Eficiencia de compresión
        - mse_reconstruction: Pérdida de información
        - acf_retention: Preservación de dependencias temporales
        - vocabulary_entropy: Distribución de tokens (solo discretos)
        - bigram_entropy: Entropía de transiciones (solo discretos)
        - token_persistence: Longitud media de runs (solo discretos)
        - top_k_coverage: Concentración top-5 tokens (solo discretos)

    Ejemplo:
        >>> metrics = evaluate_tokenization(serie, recon, 62, states)
        >>> print(metrics)
        {'compression_ratio': 16.1, 'mse_reconstruction': 0.15, ...}
    """
    T = len(original)

    # Métricas universales (aplican a todas las técnicas)
    metrics = {
        'compression_ratio': compression_ratio(T, num_tokens),
        'mse_reconstruction': mse_reconstruction(original, reconstructed),
        'acf_retention': acf_retention(original, reconstructed),
    }

    # Métricas para tokens discretos (SAX, HMM, etc.)
    if tokens is not None:
        metrics['vocabulary_entropy'] = vocabulary_entropy(tokens)
        metrics['bigram_entropy'] = bigram_entropy(tokens)
        metrics['token_persistence'] = token_persistence(tokens)
        metrics['top_k_coverage'] = top_k_coverage(tokens, k=5)

    return metrics
