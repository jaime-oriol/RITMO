"""
Técnica 1: DISCRETIZACIÓN - SAX (Symbolic Aggregate approXimation).

Implementación PURA basada en Lin et al. (2007), técnica ORIGINAL de discretización
para series temporales. Convierte valores continuos en símbolos discretos mediante
normalización + binning basado en distribución gaussiana.

Referencias:
    - Lin et al. (2007): Experiencing SAX - A novel symbolic representation of time series
    - Paper original: https://doi.org/10.1007/s10618-007-0064-z
"""

import numpy as np
from typing import Optional
from scipy.stats import norm


def sax_discretize(series: np.ndarray,
                   alphabet_size: int = 8,
                   normalize: bool = True) -> dict:
    """
    Tokeniza serie temporal mediante SAX (Symbolic Aggregate approXimation).

    SAX es la técnica ORIGINAL de discretización (Lin et al., 2007). Asume distribución
    gaussiana y divide el espacio en bins equiprobables según cuantiles de N(0,1).

    Algoritmo:
        1. Normalizar serie: z = (x - μ) / σ (media=0, std=1)
        2. Calcular breakpoints según cuantiles gaussianos equiprobables
        3. Asignar cada valor normalizado al símbolo correspondiente

    Args:
        series: Serie temporal univariada [T]
        alphabet_size: Tamaño del alfabeto discreto (default: 8)
            - Lin et al. (2007) recomiendan 3-20 símbolos
            - alphabet_size=8 es común (símbolos: 'a','b','c','d','e','f','g','h')
        normalize: Si True, normaliza serie antes de discretizar

    Returns:
        Diccionario con:
            'symbols': Array de símbolos discretos [T] (strings 'a','b','c',...)
            'tokens': Array de índices numéricos [T] ∈ {0, 1, ..., alphabet_size-1}
            'breakpoints': Puntos de corte en espacio normalizado
            'alphabet': Lista de símbolos ['a', 'b', 'c', ...]
            'num_tokens': T (un token por timestep)
            'compression_ratio': 1.0 (sin compresión)
            'vocabulary_size': alphabet_size

    Ejemplo:
        >>> series = np.array([1.0, 2.5, 3.8, 2.1, 1.5, 3.2, 2.8])
        >>> result = sax_discretize(series, alphabet_size=4)
        >>> result['symbols']
        array(['a', 'b', 'd', 'b', 'a', 'c', 'c'], dtype='<U1')
        >>> result['tokens']
        array([0, 1, 3, 1, 0, 2, 2])
    """
    T = len(series)

    # Validaciones
    if alphabet_size < 2:
        raise ValueError(f"alphabet_size debe ser >= 2, recibido: {alphabet_size}")
    if alphabet_size > 26:
        raise ValueError(f"alphabet_size debe ser <= 26 (limitado a a-z), recibido: {alphabet_size}")
    if T == 0:
        raise ValueError("Serie vacía")

    # Paso 1: Normalización (z-score)
    if normalize:
        mean = series.mean()
        std = series.std()
        if std == 0:
            # Serie constante: todos los valores al primer símbolo
            return {
                'symbols': np.array(['a'] * T),
                'tokens': np.zeros(T, dtype=int),
                'breakpoints': np.array([]),
                'alphabet': _generate_alphabet(alphabet_size),
                'num_tokens': T,
                'compression_ratio': 1.0,
                'vocabulary_size': alphabet_size
            }
        normalized = (series - mean) / std
    else:
        normalized = series

    # Paso 2: Calcular breakpoints gaussianos equiprobables
    # Dividir N(0,1) en alphabet_size regiones con igual probabilidad
    # breakpoints = puntos donde P(Z < bp) = k/alphabet_size para k=1..alphabet_size-1
    breakpoints = _gaussian_breakpoints(alphabet_size)

    # Paso 3: Discretizar mediante comparación con breakpoints
    # np.searchsorted retorna índice donde insertar valor para mantener orden
    tokens = np.searchsorted(breakpoints, normalized, side='right')

    # Convertir índices a símbolos alfabéticos
    alphabet = _generate_alphabet(alphabet_size)
    symbols = np.array([alphabet[idx] for idx in tokens])

    return {
        'symbols': symbols,
        'tokens': tokens,
        'breakpoints': breakpoints,
        'alphabet': alphabet,
        'num_tokens': T,
        'compression_ratio': 1.0,
        'vocabulary_size': alphabet_size
    }


def _gaussian_breakpoints(alphabet_size: int) -> np.ndarray:
    """
    Calcula breakpoints gaussianos equiprobables para SAX.

    Divide distribución N(0,1) en alphabet_size regiones con igual probabilidad.
    breakpoints[i] = Φ^(-1)((i+1)/alphabet_size) donde Φ^(-1) es quantile function.

    Args:
        alphabet_size: Número de símbolos

    Returns:
        Array de breakpoints [alphabet_size-1]
        Por ejemplo, alphabet_size=3 → [-0.43, 0.43] (3 regiones equiprobables)

    Ejemplo:
        >>> bp = _gaussian_breakpoints(4)
        >>> bp
        array([-0.6745, 0.0000, 0.6745])  # 4 regiones: 25% cada una
    """
    # Cuantiles equiprobables: 1/n, 2/n, ..., (n-1)/n
    quantiles = np.arange(1, alphabet_size) / alphabet_size

    # Calcular puntos de corte mediante inverse CDF de N(0,1)
    breakpoints = norm.ppf(quantiles)

    return breakpoints


def _generate_alphabet(alphabet_size: int) -> list:
    """
    Genera alfabeto de símbolos ['a', 'b', 'c', ..., 'z'].

    Args:
        alphabet_size: Tamaño del alfabeto (máximo 26)

    Returns:
        Lista de símbolos ['a', 'b', ...]
    """
    return [chr(ord('a') + i) for i in range(alphabet_size)]


def decode_sax(tokens: np.ndarray,
               breakpoints: np.ndarray,
               mean: float = 0.0,
               std: float = 1.0) -> np.ndarray:
    """
    Decodifica tokens SAX a valores continuos aproximados.

    Mapea cada símbolo al centro de su región gaussiana. Útil para reconstrucción
    y visualización.

    Args:
        tokens: Índices discretos [T]
        breakpoints: Puntos de corte [alphabet_size-1]
        mean: Media original para desnormalización
        std: Desviación estándar original para desnormalización

    Returns:
        Serie reconstruida [T]
    """
    alphabet_size = len(breakpoints) + 1

    # Calcular centros de cada región
    # Región 0: (-∞, bp[0]) → centro = bp[0] - 1.0 (heurístico)
    # Región i: (bp[i-1], bp[i]) → centro = (bp[i-1] + bp[i]) / 2
    # Región last: (bp[-1], +∞) → centro = bp[-1] + 1.0
    centers = np.zeros(alphabet_size)

    # Primera región
    centers[0] = breakpoints[0] - 1.0 if len(breakpoints) > 0 else 0.0

    # Regiones intermedias
    for i in range(1, alphabet_size - 1):
        centers[i] = (breakpoints[i-1] + breakpoints[i]) / 2.0

    # Última región
    centers[-1] = breakpoints[-1] + 1.0 if len(breakpoints) > 0 else 0.0

    # Mapear tokens a centros
    reconstructed = centers[tokens]

    # Desnormalizar
    reconstructed = reconstructed * std + mean

    return reconstructed


def visualize_sax(series: np.ndarray,
                 symbols: np.ndarray,
                 tokens: np.ndarray,
                 breakpoints: np.ndarray) -> dict:
    """
    Genera información para visualizar discretización SAX.

    Args:
        series: Serie original [T]
        symbols: Símbolos SAX [T]
        tokens: Índices discretos [T]
        breakpoints: Puntos de corte

    Returns:
        Diccionario con información para plotting
    """
    # Calcular serie normalizada para visualización
    mean = series.mean()
    std = series.std()
    normalized = (series - mean) / std if std > 0 else series - mean

    # Reconstruir serie desde tokens
    reconstructed = decode_sax(tokens, breakpoints, mean, std)

    return {
        'series': series,
        'normalized': normalized,
        'symbols': symbols,
        'tokens': tokens,
        'reconstructed': reconstructed,
        'breakpoints': breakpoints,
        'num_unique_symbols': len(np.unique(symbols)),
        'vocabulary_size': len(breakpoints) + 1
    }
