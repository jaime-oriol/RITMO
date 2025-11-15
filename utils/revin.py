"""
RevIN (Reversible Instance Normalization) para RITMO.

Implementa normalización robusta para series temporales univariadas
siguiendo Kim et al. 2022. Normaliza sobre la dimensión temporal
por instancia, permitiendo desnormalización exacta.
"""

import numpy as np
import torch
from typing import Dict, Tuple
from layers.StandardNorm import Normalize


class RevINNormalizer:
    """
    Normalizador RevIN para datasets de series temporales.

    Gestiona normalización independiente para train/val/test siguiendo
    el principio de Single Responsibility. Cada split tiene su propia
    instancia de Normalize para evitar contaminación de estadísticas.

    Args:
        num_features: Número de features (1 para univariado)
        eps: Epsilon para estabilidad numérica
        affine: Si True, aprende transformación afín
    """

    def __init__(self, num_features: int = 1, eps: float = 1e-5, affine: bool = False):
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        # Una instancia separada por split (Fail Fast si se usa antes de fit)
        self._normalizers = {}
        self._fitted = False

    def fit_transform(self,
                      train_data: np.ndarray,
                      val_data: np.ndarray = None,
                      test_data: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Ajusta normalizadores y transforma datos.

        Args:
            train_data: Serie temporal train [T] o [T, C]
            val_data: Serie temporal val (opcional)
            test_data: Serie temporal test (opcional)

        Returns:
            Dict con claves 'train', 'val', 'test' (si existen) conteniendo
            datos normalizados como arrays numpy [T] o [T, C]

        Raises:
            ValueError: Si train_data está vacío o tiene dimensiones incorrectas
        """
        if train_data is None or len(train_data) == 0:
            raise ValueError("train_data no puede estar vacío")

        # Normalizar train
        train_norm = self._normalize_split(train_data, 'train')
        result = {'train': train_norm}

        # Normalizar val si existe
        if val_data is not None and len(val_data) > 0:
            result['val'] = self._normalize_split(val_data, 'val')

        # Normalizar test si existe
        if test_data is not None and len(test_data) > 0:
            result['test'] = self._normalize_split(test_data, 'test')

        self._fitted = True
        return result

    def inverse_transform(self, data: np.ndarray, split: str = 'train') -> np.ndarray:
        """
        Desnormaliza datos usando estadísticas del split correspondiente.

        Args:
            data: Datos normalizados [T] o [T, C]
            split: Split a usar ('train', 'val', 'test')

        Returns:
            Datos desnormalizados [T] o [T, C]

        Raises:
            ValueError: Si no se ha llamado a fit_transform antes
            KeyError: Si el split no existe
        """
        if not self._fitted:
            raise ValueError("Debe llamar a fit_transform antes de inverse_transform")

        if split not in self._normalizers:
            raise KeyError(f"Split '{split}' no existe. Disponibles: {list(self._normalizers.keys())}")

        # Convertir a tensor [1, T, C]
        data_tensor = self._to_tensor(data)

        # Desnormalizar
        denorm_tensor = self._normalizers[split](data_tensor, mode='denorm')

        # Convertir a numpy
        return denorm_tensor.squeeze().detach().numpy()

    def get_statistics(self, split: str = 'train') -> Dict[str, float]:
        """
        Obtiene estadísticas de normalización del split.

        Args:
            split: Split del que obtener estadísticas

        Returns:
            Dict con 'mean' y 'stdev'

        Raises:
            ValueError: Si no se ha llamado a fit_transform
            KeyError: Si el split no existe
        """
        if not self._fitted:
            raise ValueError("Debe llamar a fit_transform antes")

        if split not in self._normalizers:
            raise KeyError(f"Split '{split}' no existe")

        normalizer = self._normalizers[split]
        return {
            'mean': normalizer.mean.item(),
            'stdev': normalizer.stdev.item()
        }

    def validate_reconstruction(self,
                               original: np.ndarray,
                               normalized: np.ndarray,
                               split: str = 'train',
                               threshold: float = 1e-6) -> Tuple[bool, float]:
        """
        Valida que norm → denorm recupera datos originales.

        Args:
            original: Datos originales [T] o [T, C]
            normalized: Datos normalizados [T] o [T, C]
            split: Split usado para normalización
            threshold: Umbral de MSE aceptable

        Returns:
            Tupla (success: bool, mse: float)
        """
        reconstructed = self.inverse_transform(normalized, split=split)
        mse = np.mean((original - reconstructed) ** 2)
        return mse < threshold, mse

    def _normalize_split(self, data: np.ndarray, split: str) -> np.ndarray:
        """Normaliza un split creando su propia instancia de Normalize."""
        # Crear instancia dedicada para este split
        normalizer = Normalize(
            num_features=self.num_features,
            eps=self.eps,
            affine=self.affine
        )

        # Convertir a tensor [1, T, C]
        data_tensor = self._to_tensor(data)

        # Normalizar
        norm_tensor = normalizer(data_tensor, mode='norm')

        # Guardar normalizer
        self._normalizers[split] = normalizer

        # Convertir a numpy
        return norm_tensor.squeeze().detach().numpy()

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Convierte numpy array a tensor [1, T, C]."""
        if data.ndim == 1:
            # [T] → [1, T, 1]
            return torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        elif data.ndim == 2:
            # [T, C] → [1, T, C]
            return torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        else:
            raise ValueError(f"Data debe tener dimensión 1 o 2, tiene {data.ndim}")
