"""
Técnica 5: FOUNDATION MODELS - Masked patches para pre-training.

Implementación PURA basada en MOMENT (Goswami et al., 2024), técnica que usa
masked patch reconstruction para pre-entrenar foundation models.

Referencias:
    - Goswami et al. (2024): MOMENT - A Family of Open Time-series Foundation Models
    - Repo oficial: https://github.com/moment-timeseries-foundation-model/moment
    - Paper: https://arxiv.org/abs/2402.03885
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


def foundation_tokenize(series: np.ndarray,
                       patch_len: int = 16,
                       stride: int = 16,
                       mask_ratio: float = 0.3,
                       random_seed: Optional[int] = None) -> dict:
    """
    Tokeniza serie temporal mediante patching + masking aleatorio estilo MOMENT.

    Técnica ORIGINAL de foundation models (Goswami et al., 2024). Combina:
    1. Segmentación en patches (como PatchTST)
    2. Masking aleatorio de patches para pre-training self-supervised

    Durante pre-training, modelo aprende a reconstruir patches enmascarados,
    generando representaciones útiles para downstream tasks.

    Args:
        series: Serie temporal univariada [T]
        patch_len: Longitud de cada patch (default: 16, MOMENT estándar)
        stride: Paso entre patches (default: 16, non-overlapping)
        mask_ratio: Proporción de patches a enmascarar (default: 0.3 = 30%)
            - MOMENT usa 30% durante pre-training
        random_seed: Semilla para reproducibilidad del masking

    Returns:
        Diccionario con:
            'patches': Array de patches [num_patches, patch_len]
            'mask': Array booleano [num_patches] (True = enmascarado)
            'masked_indices': Índices de patches enmascarados
            'visible_indices': Índices de patches visibles
            'num_patches': Número total de patches
            'num_masked': Número de patches enmascarados
            'compression_ratio': T / num_patches
            'mask_ratio_actual': Proporción real enmascarada

    Ejemplo:
        >>> series = np.arange(100)  # [0, 1, 2, ..., 99]
        >>> result = foundation_tokenize(series, patch_len=16, stride=16, mask_ratio=0.3)
        >>> result['num_patches']
        6  # 96 timesteps útiles → 6 patches de 16
        >>> result['num_masked']
        2  # 30% de 6 ≈ 2 patches enmascarados
        >>> result['mask']
        array([False, True, False, True, False, False])  # Ejemplo
    """
    T = len(series)

    # Validaciones
    if patch_len <= 0:
        raise ValueError(f"patch_len debe ser > 0, recibido: {patch_len}")
    if stride <= 0:
        raise ValueError(f"stride debe ser > 0, recibido: {stride}")
    if mask_ratio < 0 or mask_ratio > 1:
        raise ValueError(f"mask_ratio debe estar en [0, 1], recibido: {mask_ratio}")
    if T < patch_len:
        raise ValueError(f"Serie muy corta (T={T}) para patch_len={patch_len}")

    # Paso 1: Patching (IDÉNTICO a PatchTST)
    # Convertir a torch tensor [1, 1, T] para usar unfold
    x = torch.from_numpy(series).float().view(1, 1, -1)

    # Operación unfold: segmenta en patches
    patches = x.unfold(dimension=-1, size=patch_len, step=stride)
    # patches: [1, 1, num_patches, patch_len]

    # Reshape a [num_patches, patch_len]
    patches = patches.squeeze().numpy()
    if patches.ndim == 1:
        patches = patches.reshape(1, -1)

    num_patches = patches.shape[0]

    # Paso 2: Masking aleatorio
    # Usar RandomState local para no modificar el estado global del RNG
    rng = np.random.RandomState(random_seed)

    # Calcular número de patches a enmascarar
    num_masked = int(np.round(num_patches * mask_ratio))
    num_masked = max(0, min(num_masked, num_patches))  # Clamp [0, num_patches]

    # Generar máscara aleatoria
    # mask[i] = True si patch i está enmascarado
    mask_indices = rng.choice(num_patches, size=num_masked, replace=False)
    mask = np.zeros(num_patches, dtype=bool)
    mask[mask_indices] = True

    # Índices de patches visibles
    visible_indices = np.where(~mask)[0]
    masked_indices = np.where(mask)[0]

    return {
        'patches': patches,
        'mask': mask,
        'masked_indices': masked_indices,
        'visible_indices': visible_indices,
        'num_patches': num_patches,
        'num_masked': num_masked,
        'compression_ratio': T / num_patches,
        'mask_ratio_actual': num_masked / num_patches if num_patches > 0 else 0.0
    }


def apply_masking(patches: np.ndarray,
                 mask: np.ndarray,
                 mask_value: float = 0.0) -> np.ndarray:
    """
    Aplica máscara a patches, reemplazando enmascarados con valor especial.

    En MOMENT, patches enmascarados se reemplazan con embedding learnable [MASK].
    Aquí simplificamos usando valor constante para visualización.

    Args:
        patches: Array de patches [num_patches, patch_len]
        mask: Máscara booleana [num_patches]
        mask_value: Valor para patches enmascarados (default: 0.0)

    Returns:
        Patches con masking aplicado [num_patches, patch_len]
    """
    masked_patches = patches.copy()
    masked_patches[mask] = mask_value
    return masked_patches


def reconstruct_from_patches(patches: np.ndarray,
                             patch_len: int,
                             stride: int,
                             original_length: int) -> np.ndarray:
    """
    Reconstruye serie temporal desde patches.

    Para patches non-overlapping (stride=patch_len), es concatenación directa.
    Para patches solapados (stride<patch_len), promedia regiones solapadas.

    Args:
        patches: Array de patches [num_patches, patch_len]
        patch_len: Longitud de cada patch
        stride: Paso entre patches
        original_length: Longitud de serie original

    Returns:
        Serie reconstruida [T]
    """
    num_patches = patches.shape[0]

    if stride == patch_len:
        # Non-overlapping: concatenación simple
        reconstructed = patches.flatten()
        # Truncar/pad a longitud original
        if len(reconstructed) > original_length:
            reconstructed = reconstructed[:original_length]
        elif len(reconstructed) < original_length:
            pad_len = original_length - len(reconstructed)
            reconstructed = np.concatenate([reconstructed, np.zeros(pad_len)])
    else:
        # Overlapping: promediar regiones solapadas
        reconstructed = np.zeros(original_length)
        counts = np.zeros(original_length)

        for i, patch in enumerate(patches):
            start = i * stride
            end = start + patch_len
            if end > original_length:
                patch = patch[:original_length - start]
                end = original_length

            reconstructed[start:end] += patch
            counts[start:end] += 1

        # Promediar
        counts[counts == 0] = 1  # Evitar división por cero
        reconstructed /= counts

    return reconstructed


def visualize_foundation(series: np.ndarray,
                        patches: np.ndarray,
                        mask: np.ndarray,
                        patch_len: int,
                        stride: int) -> dict:
    """
    Genera información para visualizar masked patches foundation model.

    Args:
        series: Serie original [T]
        patches: Patches [num_patches, patch_len]
        mask: Máscara booleana [num_patches]
        patch_len: Longitud de patch
        stride: Paso entre patches

    Returns:
        Diccionario con información para plotting
    """
    # Aplicar masking para visualización
    masked_patches = apply_masking(patches, mask, mask_value=0.0)

    # Reconstruir serie desde patches originales
    reconstructed_original = reconstruct_from_patches(patches, patch_len, stride, len(series))

    # Reconstruir serie desde patches enmascarados
    reconstructed_masked = reconstruct_from_patches(masked_patches, patch_len, stride, len(series))

    # Calcular posiciones de patches en serie original
    patch_positions = [i * stride for i in range(len(patches))]

    return {
        'series': series,
        'patches': patches,
        'masked_patches': masked_patches,
        'mask': mask,
        'reconstructed_original': reconstructed_original,
        'reconstructed_masked': reconstructed_masked,
        'patch_positions': patch_positions,
        'num_masked': mask.sum(),
        'mask_ratio': mask.mean()
    }
