"""
Generador de embeddings estructurados desde estados HMM.

Implementa transformación tokens discretos → embeddings vectoriales
e_k = [μ_k, σ_k, A[k,:]] preservando interpretabilidad estadística.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class EmbeddingGenerator(nn.Module):
    """
    Genera embeddings estructurados desde tokens HMM.

    Arquitectura:
        tokens[T] → lookup_table[K, 2+K] → embeddings[T, 2+K]
                  → projection[2+K, d_model] → projected[T, d_model]
                  → + positional_encoding → ready_for_transformer

    Parámetros FROZEN (no gradientes):
        - mu, sigma, A: Vienen del HMM entrenado

    Parámetros LEARNABLE (con gradientes):
        - projection layer: aprende representación óptima

    Args:
        hmm_params: Dict con claves 'A', 'mu', 'sigma' (numpy arrays)
        d_model: Dimensión de salida para Transformer (default: 128)
        device: 'cpu' o 'cuda'

    Ejemplo:
        >>> from hmm import load_hmm_params
        >>> params = load_hmm_params('./cache/hmm_etth1_K5.pth')
        >>> emb_gen = EmbeddingGenerator(params, d_model=128)
        >>> tokens = np.array([0, 1, 2, 1, 0])  # [T]
        >>> embeddings = emb_gen(tokens)  # [T, d_model]
    """

    def __init__(self,
                 hmm_params: Dict[str, np.ndarray],
                 d_model: int = 128,
                 device: str = 'cpu'):
        super(EmbeddingGenerator, self).__init__()

        # Validaciones
        required_keys = {'A', 'mu', 'sigma'}
        if not required_keys.issubset(hmm_params.keys()):
            missing = required_keys - hmm_params.keys()
            raise ValueError(f"hmm_params falta claves: {missing}")

        self.device = device
        K = len(hmm_params['mu'])
        self.K = K
        self.embedding_dim = 2 + K  # [mu, sigma, A[k,:]]
        self.d_model = d_model

        # Crear lookup table: embedding_table[k] = [μ_k, σ_k, A[k,:]]
        embedding_table = np.zeros((K, self.embedding_dim))
        for k in range(K):
            embedding_table[k] = np.concatenate([
                [hmm_params['mu'][k]],
                [hmm_params['sigma'][k]],
                hmm_params['A'][k, :]
            ])

        # Registrar como buffer (FROZEN, no gradientes)
        self.register_buffer(
            'embedding_table',
            torch.from_numpy(embedding_table).float()
        )

        # Projection layer: (2+K) → d_model (LEARNABLE)
        self.projection = nn.Linear(self.embedding_dim, d_model)

    def forward(self, tokens: np.ndarray) -> torch.Tensor:
        """
        Genera embeddings desde tokens.

        Args:
            tokens: Array de estados [T] con valores en {0, ..., K-1}

        Returns:
            Embeddings proyectados [T, d_model]

        Raises:
            ValueError: Si tokens contiene valores fuera de rango [0, K-1]
        """
        # Validaciones
        if np.any(tokens < 0) or np.any(tokens >= self.K):
            raise ValueError(f"tokens debe estar en [0, {self.K-1}], "
                           f"recibido: min={tokens.min()}, max={tokens.max()}")

        # Convertir a tensor
        tokens_tensor = torch.from_numpy(tokens).long().to(self.device)

        # Lookup: tokens[T] → embeddings[T, 2+K]
        embeddings_raw = self.embedding_table[tokens_tensor]

        # Projection: [T, 2+K] → [T, d_model]
        embeddings_projected = self.projection(embeddings_raw)

        return embeddings_projected

    def get_embedding_table(self) -> torch.Tensor:
        """
        Retorna tabla de embeddings crudos (sin proyección).

        Returns:
            Tensor [K, 2+K] con embeddings estructurados
        """
        return self.embedding_table

    def get_state_info(self, state: int) -> Dict[str, float]:
        """
        Obtiene información estadística de un estado.

        Args:
            state: Índice de estado [0, K-1]

        Returns:
            Dict con 'mu', 'sigma', 'transitions'
        """
        if state < 0 or state >= self.K:
            raise ValueError(f"state debe estar en [0, {self.K-1}]")

        emb = self.embedding_table[state].numpy()
        return {
            'mu': float(emb[0]),
            'sigma': float(emb[1]),
            'transitions': emb[2:].tolist()
        }
