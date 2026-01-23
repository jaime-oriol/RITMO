"""
Normalización reversible (RevIN) para series temporales.
Idea: Normalizar entrada, procesar, desnormalizar salida.
Esto ayuda a manejar series no estacionarias (con media/varianza cambiante).
"""

import torch  # Framework de deep learning
import torch.nn as nn  # Módulo de redes neuronales


class Normalize(nn.Module):
    """
    RevIN (Reversible Instance Normalization).
    Normaliza cada instancia individualmente y puede revertir la operación.
    Útil para series temporales con distribution shift.

    Modos:
    - 'norm': normaliza (entrada del modelo)
    - 'denorm': desnormaliza (salida del modelo)
    """
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        Args:
            num_features: Número de canales/features
            eps: Epsilon para estabilidad numérica
            affine: Si True, aprende parámetros de escala y sesgo
            subtract_last: Si True, resta el último valor en vez de la media
            non_norm: Si True, no hace nada (bypass)
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm

        # Parámetros afines opcionales
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        """
        x: tensor a normalizar/desnormalizar
        mode: 'norm' o 'denorm'
        """
        if mode == 'norm':
            self._get_statistics(x)  # Calcular media y std
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        """Inicializa parámetros afines: peso=1, sesgo=0."""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """
        Calcula estadísticas de la entrada.
        Se guardan para usar en desnormalización posterior.
        """
        # Reducir todas las dimensiones excepto batch y features
        dim2reduce = tuple(range(1, x.ndim - 1))

        if self.subtract_last:
            # Usar último valor como referencia (para series trending)
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            # Usar media como referencia
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

        # Desviación estándar (siempre se calcula)
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        """Normaliza: (x - centro) / std [* peso + sesgo]."""
        if self.non_norm:
            return x  # Bypass

        # Centrar (restar media o último valor)
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean

        # Escalar por desviación estándar
        x = x / self.stdev

        # Parámetros afines opcionales
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        """Desnormaliza: invierte la normalización."""
        if self.non_norm:
            return x  # Bypass

        # Revertir parámetros afines
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)

        # Revertir escalado
        x = x * self.stdev

        # Revertir centrado
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
