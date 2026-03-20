"""
Métricas intrínsecas para evaluar tokenización SIN tareas downstream.
Permiten comparar técnicas de tokenización usando solo los tokens generados.
No requieren entrenar modelos ni ejecutar forecasting/clasificación.

Referencias:
    - Chiarot & Silvestri (2022): Time Series Compression Survey, ACM Computing Surveys, 55(10).
    - Uzan et al. (2024): Greed is All You Need, ACL 2024.
    - Levenshtein (1966): Binary codes capable of correcting deletions, insertions, and reversals,
      Soviet Physics Doklady, 10(8), pp. 707-710.
"""

import numpy as np  # Operaciones matemáticas con arrays
from collections import Counter  # Para contar ocurrencias de bigramas
from scipy import stats  # Para calcular entropía
from typing import Dict, Any, Callable, List  # Anotaciones de tipos


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

    Ref: Métrica estándar de procesamiento de señales (Box & Jenkins, 1976).
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


def perturbation_stability(series: np.ndarray,
                           tokenize_and_reconstruct: Callable[[np.ndarray], np.ndarray],
                           noise_levels: List[float] = None,
                           n_reps: int = 3,
                           random_seed: int = 42) -> Dict[str, Any]:
    """
    Estabilidad ante perturbaciones gaussianas (versión ligera).

    Añade ruido gaussiano a la serie, re-tokeniza y mide cuánto cambian
    MSE de reconstrucción y ACF retention respecto a la versión sin ruido.

    Protocolo:
        - 2 niveles de ruido (bajo=0.1*std, medio=0.5*std)
        - 3 repeticiones por nivel
        - Reportar cambio relativo vs baseline sin ruido

    Args:
        series: Serie temporal normalizada [T]
        tokenize_and_reconstruct: Callback que tokeniza y reconstruye.
            Recibe np.ndarray [T], devuelve np.ndarray [T].
            Ejemplo para SAX:
                lambda s: decode_sax(sax_discretize(s)['tokens'],
                                     sax_discretize(s)['breakpoints'])
        noise_levels: Niveles de ruido como fracción de std (default: [0.1, 0.5])
        n_reps: Repeticiones por nivel (default: 3)
        random_seed: Semilla para reproducibilidad (default: 42)

    Returns:
        Diccionario con:
            'baseline_mse': MSE sin ruido
            'baseline_acf': ACF retention sin ruido
            'by_level': dict por nivel con media de cambio relativo en MSE y ACF
            'summary': cambio relativo medio global
    """
    if noise_levels is None:
        noise_levels = [0.1, 0.5]
    rng = np.random.RandomState(random_seed)
    std = series.std()
    if std == 0:
        std = 1.0

    # Baseline sin ruido
    recon_clean = tokenize_and_reconstruct(series)
    baseline_mse = mse_reconstruction(series, recon_clean)
    baseline_acf = acf_retention(series, recon_clean)

    by_level = {}
    for level in noise_levels:
        mse_changes = []
        acf_changes = []

        for _ in range(n_reps):
            noisy = series + rng.normal(0, level * std, len(series))
            recon_noisy = tokenize_and_reconstruct(noisy)

            noisy_mse = mse_reconstruction(noisy, recon_noisy)
            noisy_acf = acf_retention(noisy, recon_noisy)

            # Cambio relativo: (perturbado - baseline) / baseline
            if baseline_mse > 0:
                mse_changes.append((noisy_mse - baseline_mse) / baseline_mse)
            else:
                mse_changes.append(noisy_mse)

            if baseline_acf > 0:
                acf_changes.append((noisy_acf - baseline_acf) / baseline_acf)
            else:
                acf_changes.append(noisy_acf)

        by_level[level] = {
            'mse_change_rel': float(np.mean(mse_changes)),
            'acf_change_rel': float(np.mean(acf_changes)),
            'mse_change_std': float(np.std(mse_changes)),
            'acf_change_std': float(np.std(acf_changes)),
        }

    # Resumen global
    all_mse = [by_level[l]['mse_change_rel'] for l in noise_levels]
    all_acf = [by_level[l]['acf_change_rel'] for l in noise_levels]

    return {
        'baseline_mse': float(baseline_mse),
        'baseline_acf': float(baseline_acf),
        'by_level': by_level,
        'summary': {
            'avg_mse_change_rel': float(np.mean(all_mse)),
            'avg_acf_change_rel': float(np.mean(all_acf)),
        }
    }


def edit_distance_normalized(tokens_a: np.ndarray, tokens_b: np.ndarray) -> float:
    """
    Distancia de edición normalizada entre dos secuencias de tokens discretos.

    Mide cuántas operaciones (inserción, eliminación, sustitución) se necesitan
    para transformar una secuencia en otra, normalizado por la longitud máxima.

    Comparación intra-técnica: tokens de serie original vs perturbada.

    Args:
        tokens_a: Secuencia de tokens discretos [T1]
        tokens_b: Secuencia de tokens discretos [T2]

    Returns:
        Distancia normalizada [0, 1]:
        - 0.0 = secuencias idénticas
        - 1.0 = completamente diferentes

    Ejemplo:
        >>> a = np.array([0, 1, 2, 3, 4])
        >>> b = np.array([0, 1, 3, 3, 4])
        >>> edit_distance_normalized(a, b)
        0.2  # 1 sustitución / 5
    """
    n, m = len(tokens_a), len(tokens_b)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0

    # Levenshtein con DP optimizado en memoria (2 filas)
    prev = np.arange(m + 1, dtype=int)
    curr = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if tokens_a[i - 1] == tokens_b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # Eliminación
                curr[j - 1] + 1,  # Inserción
                prev[j - 1] + cost  # Sustitución
            )
        prev, curr = curr, prev

    return float(prev[m]) / max(n, m)


def token_distance_continuous(tokens_a: np.ndarray,
                              tokens_b: np.ndarray,
                              metric: str = 'l2') -> float:
    """
    Distancia media entre dos secuencias de tokens continuos alineados.

    Comparación intra-técnica: tokens de serie original vs perturbada.
    Requiere misma longitud y alineación.

    Args:
        tokens_a: Tokens continuos [N, D] o [N] (patches, componentes, etc.)
        tokens_b: Tokens continuos [N, D] o [N] (misma shape que tokens_a)
        metric: 'l2' (distancia euclidiana) o 'cosine' (distancia coseno)

    Returns:
        Distancia media entre pares de tokens:
        - 0.0 = tokens idénticos
        - Mayor valor = más diferentes

    Ejemplo:
        >>> a = np.array([[1, 2], [3, 4]])  # 2 patches de dim 2
        >>> b = np.array([[1, 2], [3, 5]])  # Segundo patch ligeramente diferente
        >>> token_distance_continuous(a, b, metric='l2')
        0.5  # media de [0.0, 1.0]
    """
    a = tokens_a.reshape(tokens_a.shape[0], -1) if tokens_a.ndim > 1 else tokens_a.reshape(-1, 1)
    b = tokens_b.reshape(tokens_b.shape[0], -1) if tokens_b.ndim > 1 else tokens_b.reshape(-1, 1)

    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    if metric == 'l2':
        distances = np.sqrt(np.sum((a - b) ** 2, axis=1))
    elif metric == 'cosine':
        # cosine distance = 1 - cosine_similarity
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True)
        # Evitar división por cero
        norm_a = np.where(norm_a == 0, 1, norm_a)
        norm_b = np.where(norm_b == 0, 1, norm_b)
        cosine_sim = np.sum((a / norm_a) * (b / norm_b), axis=1)
        distances = 1.0 - cosine_sim
    else:
        raise ValueError(f"metric debe ser 'l2' o 'cosine', recibido: {metric}")

    return float(np.mean(distances))


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
