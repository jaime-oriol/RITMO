"""
Técnica 4: DESCOMPOSICIÓN - Trend + Seasonal + Residual.

Implementación PURA extraída de DLinear/TimeMixer (Zeng et al., 2023; Wang et al., 2024).
Reutiliza series_decomp de layers/Autoformer_EncDec.py sin capas neuronales.

Referencias:
    - Zeng et al. (2023): Are Transformers Effective for Time Series Forecasting? (DLinear)
    - Wu et al. (2021): Autoformer (serie_decomp original)
    - Implementación original: layers/Autoformer_EncDec.py, líneas 21-54
"""

import numpy as np
import torch
import torch.nn as nn


def moving_average(series: np.ndarray, kernel_size: int = 25) -> np.ndarray:
    """
    Calcula promedio móvil para extraer tendencia (componente trend).

    Implementación PURA de layers/Autoformer_EncDec.py:moving_avg (líneas 21-39).
    Usa AvgPool1d con padding reflejado en extremos para evitar edge effects.

    Args:
        series: Serie temporal [T]
        kernel_size: Tamaño ventana promedio móvil (default: 25, igual que Autoformer)

    Returns:
        Tendencia suavizada [T] (mismo tamaño que serie original)

    Ejemplo:
        >>> series = np.sin(np.linspace(0, 4*np.pi, 100)) + np.linspace(0, 2, 100)
        >>> trend = moving_average(series, kernel_size=25)
        >>> trend.shape
        (100,)  # Mismo tamaño que serie original
    """
    T = len(series)
    if kernel_size > T:
        raise ValueError(f"kernel_size={kernel_size} no puede ser > T={T}")

    # Convertir a torch tensor [1, 1, T]
    x = torch.from_numpy(series).float().view(1, 1, -1)

    # Padding reflejando extremos (IDÉNTICO a líneas 33-35)
    pad_size = (kernel_size - 1) // 2
    front = x[:, :, 0:1].repeat(1, 1, pad_size)  # Repetir primer valor
    end = x[:, :, -1:].repeat(1, 1, pad_size)    # Repetir último valor
    x_padded = torch.cat([front, x, end], dim=-1)

    # AvgPool1d (IDÉNTICO a línea 29)
    avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
    trend = avg_pool(x_padded)

    # Convertir a numpy [T]
    trend = trend.squeeze().numpy()

    return trend


def decomposition_tokenize(series: np.ndarray,
                           kernel_size: int = 25) -> dict:
    """
    Tokeniza serie temporal mediante descomposición Trend-Seasonal.

    Implementación PURA de layers/Autoformer_EncDec.py:series_decomp (líneas 41-54).
    Descompone serie en:
        - Trend: componente de baja frecuencia (promedio móvil)
        - Seasonal: componente de alta frecuencia (residuo = serie - trend)

    Esta descomposición actúa como tokenización porque cada componente se modela
    independientemente (DLinear aplica capas lineales separadas a cada uno).

    Args:
        series: Serie temporal univariada [T]
        kernel_size: Tamaño ventana promedio móvil (default: 25, DLinear estándar)

    Returns:
        Diccionario con:
            'seasonal': componente estacional [T]
            'trend': componente de tendencia [T]
            'num_tokens': 2 (dos componentes = dos "tokens")
            'compression_ratio': T / 2

    Ejemplo:
        >>> series = np.sin(np.linspace(0, 4*np.pi, 100)) + np.linspace(0, 2, 100)
        >>> tokens = decomposition_tokenize(series, kernel_size=25)
        >>> tokens['seasonal'].shape
        (100,)
        >>> tokens['trend'].shape
        (100,)
        >>> tokens['num_tokens']
        2
    """
    # Validaciones
    if len(series) < kernel_size:
        raise ValueError(f"Serie muy corta (T={len(series)}) para kernel_size={kernel_size}")

    # Calcular trend mediante promedio móvil (IDÉNTICO a línea 51)
    trend = moving_average(series, kernel_size)

    # Seasonal = residuo (IDÉNTICO a línea 52)
    seasonal = series - trend

    return {
        'seasonal': seasonal,
        'trend': trend,
        'num_tokens': 2,  # Dos componentes = dos "tokens"
        'compression_ratio': len(series) / 2.0
    }


def visualize_decomposition(tokens: dict) -> dict:
    """
    Genera información para visualizar descomposición.

    Args:
        tokens: Output de decomposition_tokenize()

    Returns:
        Diccionario con información para plotting
    """
    seasonal = tokens['seasonal']
    trend = tokens['trend']
    reconstruction = seasonal + trend  # Serie reconstruida

    return {
        'seasonal': seasonal,
        'trend': trend,
        'reconstruction': reconstruction,
        'num_tokens': tokens['num_tokens'],
        'compression_ratio': tokens['compression_ratio']
    }
