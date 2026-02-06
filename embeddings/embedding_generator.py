"""
Generador de embeddings desde estados HMM.
Convierte cada estado (0, 1, 2...) en un vector con información útil:
- Media del estado (mu)
- Desviación del estado (sigma)
- Probabilidades de transición a otros estados
"""

import torch  # Framework de deep learning
import torch.nn as nn  # Módulo de redes neuronales
import numpy as np  # Operaciones matemáticas
from typing import Dict  # Para indicar tipos de datos


class EmbeddingGenerator(nn.Module):
    """
    Convierte tokens HMM en embeddings vectoriales para usar en Transformers.

    Flujo:
        1. Cada token (0, 1, 2...) se busca en una tabla de embeddings
        2. El embedding contiene [mu, sigma, probabilidades de transición]
        3. Una capa lineal proyecta al tamaño que necesita el Transformer

    La tabla de embeddings viene del HMM entrenado (no se modifica).
    Solo la capa de proyección aprende durante el entrenamiento.

    Entrada al crear:
        hmm_params: Parámetros del HMM (de load_hmm_params)
        d_model: Tamaño de salida para el Transformer (ej: 128)
        device: 'cpu' o 'cuda' para GPU
    """

    def __init__(self,
                 hmm_params: Dict[str, np.ndarray],
                 d_model: int = 128,
                 device: str = 'cpu'):
        """Inicializa el generador de embeddings."""
        super().__init__()

        # Verificar que tenemos todos los parámetros necesarios
        required_keys = {'A', 'mu', 'sigma'}
        if not required_keys.issubset(hmm_params.keys()):
            missing = required_keys - hmm_params.keys()
            raise ValueError(f"hmm_params falta claves: {missing}")

        # Guardar configuración
        self.device = device

        # Convertir parámetros a numpy si son tensores
        if isinstance(hmm_params['mu'], torch.Tensor):
            mu = hmm_params['mu'].cpu().numpy()
            sigma = hmm_params['sigma'].cpu().numpy()
            A = hmm_params['A'].cpu().numpy()
        else:
            mu = hmm_params['mu']
            sigma = hmm_params['sigma']
            A = hmm_params['A']

        K = len(mu)  # Número de estados
        self.K = K
        self.embedding_dim = 2 + K  # Tamaño: mu + sigma + K transiciones
        self.d_model = d_model

        # Construir tabla de embeddings: una fila por estado
        # Cada fila = [media, desviación, prob_transición_a_estado_0, ..., prob_transición_a_estado_K-1]
        embedding_table = np.zeros((K, self.embedding_dim))
        for k in range(K):
            embedding_table[k] = np.concatenate([
                [mu[k]],      # Media del estado k
                [sigma[k]],   # Desviación del estado k
                A[k, :]       # Fila k de matriz de transición
            ])

        # Guardar tabla como "buffer" (no se entrena, viene del HMM)
        self.register_buffer(
            'embedding_table',
            torch.from_numpy(embedding_table).float()
        )

        # Capa de proyección: reduce/expande al tamaño del Transformer
        # Esta SÍ se entrena (tiene gradientes)
        self.projection = nn.Linear(self.embedding_dim, d_model)

    def forward(self, tokens) -> torch.Tensor:
        """
        Convierte tokens HARD (Viterbi) en embeddings.

        Entrada:
            tokens: Lista de estados [0, 1, 2, ...] de longitud T (numpy array o tensor)

        Salida:
            Tensor de embeddings [T, d_model] listo para el Transformer
        """
        # Convertir a tensor si es numpy array
        if isinstance(tokens, np.ndarray):
            tokens_tensor = torch.from_numpy(tokens).long().to(self.device)
        elif isinstance(tokens, torch.Tensor):
            tokens_tensor = tokens.long().to(self.device)
        else:
            raise TypeError(f"tokens debe ser np.ndarray o torch.Tensor, recibido: {type(tokens)}")

        # Verificar que los tokens estén en rango válido
        if torch.any(tokens_tensor < 0) or torch.any(tokens_tensor >= self.K):
            raise ValueError(f"tokens debe estar en [0, {self.K-1}], "
                           f"recibido: min={tokens_tensor.min()}, max={tokens_tensor.max()}")

        # Buscar cada token en la tabla de embeddings
        # tokens[T] → embeddings[T, 2+K]
        embeddings_raw = self.embedding_table[tokens_tensor]

        # Proyectar al tamaño del Transformer
        # [T, 2+K] → [T, d_model]
        embeddings_projected = self.projection(embeddings_raw)

        return embeddings_projected

    def forward_soft(self, gamma: torch.Tensor) -> torch.Tensor:
        """
        Convierte posteriors SOFT (gamma) en embeddings via mezcla ponderada.
        e_t = sum_k gamma_t(k) * e_k  (cada timestep obtiene embedding unico)

        Entrada:
            gamma: Posteriors [T, K] del forward-backward (cada fila suma ~1)

        Salida:
            Tensor de embeddings [T, d_model]
        """
        if isinstance(gamma, np.ndarray):
            gamma = torch.from_numpy(gamma).float().to(self.device)
        else:
            gamma = gamma.float().to(self.device)

        # Mezcla ponderada: [T, K] @ [K, 2+K] = [T, 2+K]
        embeddings_raw = gamma @ self.embedding_table

        # Proyectar: [T, 2+K] → [T, d_model]
        return self.projection(embeddings_raw)

    def get_embedding_table(self) -> torch.Tensor:
        """
        Devuelve la tabla completa de embeddings (sin proyectar).

        Salida:
            Tensor [K, 2+K] donde cada fila es un embedding de estado
        """
        return self.embedding_table

    def get_state_info(self, state: int) -> Dict[str, float]:
        """
        Obtiene información detallada de un estado específico.

        Entrada:
            state: Número de estado (0, 1, 2, ...)

        Salida:
            Diccionario con:
            - 'mu': Media del estado
            - 'sigma': Desviación del estado
            - 'transitions': Lista de probabilidades de transición
        """
        # Verificar que el estado existe
        if state < 0 or state >= self.K:
            raise ValueError(f"state debe estar en [0, {self.K-1}]")

        # Extraer embedding del estado
        emb = self.embedding_table[state].cpu().numpy()

        # Desempaquetar componentes
        return {
            'mu': float(emb[0]),           # Primer elemento = media
            'sigma': float(emb[1]),        # Segundo elemento = desviación
            'transitions': emb[2:].tolist()  # Resto = probabilidades de transición
        }
