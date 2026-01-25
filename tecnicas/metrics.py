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


def fertility(T_original: int, num_tokens: int) -> float:
    """
    Fertilidad: tokens por timestep.
    Métrica de NLP adaptada a series temporales.

    Entrada:
        T_original: Longitud de la serie original
        num_tokens: Número de tokens generados

    Salida:
        Tokens por timestep:
        - <1 = compresión (menos tokens que timesteps)
        - =1 = un token por timestep
        - >1 = expansión (más tokens que timesteps)

    Nota: Es el inverso de compression_ratio.

    Ejemplo:
        PatchTST: 62 tokens / 1000 timesteps = 0.062 (comprime)
        LLMTime: 10496 chars / 1000 timesteps = 10.5 (expande)

    Ref: Rust et al. 2021 (NLP)
    """
    if T_original == 0:
        return 0.0

    return num_tokens / T_original


def evaluate_tokenization(original: np.ndarray,
                          reconstructed: np.ndarray,
                          num_tokens: int,
                          tokens: np.ndarray = None) -> Dict[str, float]:
    """
    Evalúa tokenización con todas las métricas universales.
    Función de conveniencia que calcula todas las métricas a la vez.

    Entrada:
        original: Serie temporal original [T]
        reconstructed: Serie reconstruida desde tokens [T]
        num_tokens: Número de tokens generados
        tokens: Secuencia de tokens discretos (opcional, para entropy)

    Salida:
        Diccionario con todas las métricas:
        - compression_ratio: Eficiencia de compresión
        - mse_reconstruction: Pérdida de información
        - acf_retention: Preservación de dependencias
        - fertility: Tokens por timestep
        - vocabulary_entropy: Distribución de tokens (solo si tokens != None)

    Ejemplo:
        >>> metrics = evaluate_tokenization(serie, recon, 62, states)
        >>> print(metrics)
        {'compression_ratio': 16.1, 'mse_reconstruction': 0.15, ...}
    """
    T = len(original)

    # Calcular métricas universales (aplican a todas las técnicas)
    metrics = {
        'compression_ratio': compression_ratio(T, num_tokens),
        'mse_reconstruction': mse_reconstruction(original, reconstructed),
        'acf_retention': acf_retention(original, reconstructed),
        'fertility': fertility(T, num_tokens),
    }

    # Entropía solo para técnicas con tokens discretos (SAX, HMM)
    if tokens is not None:
        metrics['vocabulary_entropy'] = vocabulary_entropy(tokens)

    return metrics
