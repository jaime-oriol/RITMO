"""
Máscaras para mecanismos de atención.
Usadas para implementar atención causal y ProbSparse attention.
"""

import torch  # Deep learning


class TriangularCausalMask():
    """
    Máscara causal triangular superior.
    Impide que posición i atienda a posiciones j > i (futuro).
    Usada en decoders autoregresivos.
    """
    def __init__(self, B, L, device="cpu"):
        """
        B: Batch size
        L: Longitud de secuencia
        device: 'cpu' o 'cuda'
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # triu: triangular superior (True = enmascarar)
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """Retorna la máscara [B, 1, L, L]."""
        return self._mask


class ProbMask():
    """
    Máscara para ProbSparse attention (Informer).
    Combina máscara causal con selección de queries top-k.
    """
    def __init__(self, B, H, L, index, scores, device="cpu"):
        """
        B: Batch size
        H: Número de heads
        L: Longitud de secuencia
        index: Índices de queries seleccionadas
        scores: Puntuaciones de atención
        device: 'cpu' o 'cuda'
        """
        # Máscara causal base
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        # Expandir para batch y heads
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        # Seleccionar solo las queries en index
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        """Retorna la máscara."""
        return self._mask
