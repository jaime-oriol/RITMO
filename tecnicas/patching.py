"""
Técnica 3: PATCHING - Segmentación en ventanas fijas.

Implementación PURA extraída de PatchTST (Nie et al., 2023).
Reutiliza la operación unfold() de layers/Embed.py:PatchEmbedding sin capas de embedding.

Referencias:
    - Nie et al. (2023): A Time Series is Worth 64 Words (PatchTST)
    - Implementación original: layers/Embed.py, líneas 165-191
"""

import numpy as np
import torch
import torch.nn as nn


def patching_tokenize(series: np.ndarray,
                      patch_len: int = 16,
                      stride: int = 8) -> np.ndarray:
    """
    Tokeniza serie temporal mediante segmentación en patches fijos.

    Implementación PURA de PatchTST sin capas de embedding. Usa torch.unfold()
    que desliza ventana de tamaño patch_len con paso stride, generando patches
    solapados que actúan como tokens.

    Args:
        series: Serie temporal univariada [T]
        patch_len: Longitud de cada patch (default: 16, igual que PatchTST)
        stride: Paso entre patches (default: 8, stride típico)

    Returns:
        Patches [num_patches, patch_len] donde num_patches ≈ (T-patch_len)/stride + 1

    Ejemplo:
        >>> series = np.arange(100)  # [0, 1, 2, ..., 99]
        >>> patches = patching_tokenize(series, patch_len=16, stride=8)
        >>> patches.shape
        (11, 16)  # 11 patches de longitud 16
        >>> patches[0]  # Primer patch
        array([0, 1, 2, ..., 15])
        >>> patches[1]  # Segundo patch (desplazado +8)
        array([8, 9, 10, ..., 23])
    """
    T = len(series)

    # Validaciones
    if patch_len <= 0:
        raise ValueError(f"patch_len debe ser > 0, recibido: {patch_len}")
    if stride <= 0:
        raise ValueError(f"stride debe ser > 0, recibido: {stride}")
    if T < patch_len:
        raise ValueError(f"Serie muy corta (T={T}) para patch_len={patch_len}")

    # Convertir a torch tensor [1, 1, T] para usar unfold
    x = torch.from_numpy(series).float().view(1, 1, -1)

    # Operación unfold: IDÉNTICA a layers/Embed.py:186
    # x.unfold(dimension=-1, size=patch_len, step=stride)
    # dimension=-1: opera sobre dimensión temporal
    # size=patch_len: tamaño de cada ventana
    # step=stride: paso entre ventanas
    patches = x.unfold(dimension=-1, size=patch_len, step=stride)

    # patches: [1, 1, num_patches, patch_len]
    # Reshape a [num_patches, patch_len]
    patches = patches.squeeze().numpy()

    # Si solo hay 1 patch, asegurar shape [1, patch_len]
    if patches.ndim == 1:
        patches = patches.reshape(1, -1)

    return patches


def visualize_patches(series: np.ndarray,
                      patches: np.ndarray,
                      patch_len: int,
                      stride: int) -> dict:
    """
    Genera información para visualizar patches sobre serie original.

    Args:
        series: Serie temporal original [T]
        patches: Patches generados [num_patches, patch_len]
        patch_len: Longitud de cada patch
        stride: Paso entre patches

    Returns:
        Diccionario con:
            'series': serie original
            'patches': patches tokenizados
            'positions': posiciones inicio de cada patch en serie original
            'num_patches': número total de patches
            'compression_ratio': T / num_patches
    """
    num_patches = patches.shape[0]
    positions = [i * stride for i in range(num_patches)]

    return {
        'series': series,
        'patches': patches,
        'positions': positions,
        'num_patches': num_patches,
        'compression_ratio': len(series) / num_patches
    }
