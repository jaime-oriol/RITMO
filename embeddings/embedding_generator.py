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

        # Guardar mu y sigma como buffers para residual intra-regimen
        self.register_buffer('mu', torch.from_numpy(mu).float())
        self.register_buffer('sigma', torch.from_numpy(sigma).float())

        # Capa de proyección: reduce/expande al tamaño del Transformer
        # Esta SÍ se entrena (tiene gradientes)
        self.projection = nn.Linear(self.embedding_dim, d_model)

        # Proyección para soft_residual: r_t (1) + mu_k (1) + sigma_k (1) + A[k,:] (K) = 3 + K
        self.projection_residual = nn.Linear(3 + K, d_model)

        # Proyección para augmented: x_t (1) + gamma (K) = 1 + K
        self.projection_augmented = nn.Linear(1 + K, d_model)

        # Proyección para patched: parchea features [x_t, gamma_t] en patches
        # patch_len * (1+K) features por patch → d_model
        self.patch_len_hmm = 16  # Mismo que PatchTST
        self.projection_patched = nn.Linear(self.patch_len_hmm * (1 + K), d_model)

        # Proyección separada para split: x_t y gamma no interfieren
        # Mismo diseño que DecompositionEmbedding (trend/seasonal separados)
        self.projection_value = nn.Linear(1, d_model // 2)    # señal cruda
        self.projection_gamma = nn.Linear(K, d_model // 2)    # info regimen
        self.norm_split = nn.LayerNorm(d_model)

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

    def forward_soft_residual(self, gamma: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Soft gamma + residual intra-regimen.
        r_t = (x_t - mu_soft_t) / sigma_soft_t donde mu/sigma_soft son mezcla ponderada.
        e_t = projection([r_t, mu_soft_t, sigma_soft_t, A_soft_t])

        Entrada:
            gamma: Posteriors [T, K] del forward-backward
            x: Valores observados [T] o [T, 1]

        Salida:
            Tensor de embeddings [T, d_model]
        """
        if isinstance(gamma, np.ndarray):
            gamma = torch.from_numpy(gamma).float().to(self.device)
        else:
            gamma = gamma.float().to(self.device)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        else:
            x = x.float().to(self.device)

        if x.dim() > 1:
            x = x.squeeze(-1)  # [T, 1] -> [T]

        # mu y sigma ponderados por gamma: [T, K] @ [K] = [T]
        mu_soft = gamma @ self.mu          # [T]
        sigma_soft = gamma @ self.sigma    # [T]
        sigma_soft = torch.clamp(sigma_soft, min=1e-6)

        # Residual intra-regimen
        r_t = (x - mu_soft) / sigma_soft   # [T]

        # Transiciones ponderadas: [T, K] @ [K, K] = [T, K]
        A_soft = gamma @ self.embedding_table[:, 2:]

        # Concatenar: [r_t, mu_soft, sigma_soft, A_soft] -> [T, 3+K]
        features = torch.cat([
            r_t.unsqueeze(-1),          # [T, 1]
            mu_soft.unsqueeze(-1),      # [T, 1]
            sigma_soft.unsqueeze(-1),   # [T, 1]
            A_soft,                     # [T, K]
        ], dim=-1)

        return self.projection_residual(features)

    def forward_augmented(self, gamma: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Valor crudo + gamma posteriors: el HMM ENRIQUECE la serie, no la reemplaza.
        e_t = projection([x_t, gamma_t(0), ..., gamma_t(K-1)])

        El valor x_t preserva TODA la informacion de la serie original.
        gamma_t aporta la probabilidad de estar en cada regimen, codificando
        estructura temporal explicita aprendida por el HMM.

        Entrada:
            gamma: Posteriors [T, K] del forward-backward
            x: Valores observados [T] o [T, 1]

        Salida:
            Tensor de embeddings [T, d_model]
        """
        if isinstance(gamma, np.ndarray):
            gamma = torch.from_numpy(gamma).float().to(self.device)
        else:
            gamma = gamma.float().to(self.device)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        else:
            x = x.float().to(self.device)

        if x.dim() > 1:
            x = x.squeeze(-1)  # [T, 1] -> [T]

        # Concatenar: [x_t, gamma_t] -> [T, 1+K]
        features = torch.cat([
            x.unsqueeze(-1),  # [T, 1]
            gamma,            # [T, K]
        ], dim=-1)

        return self.projection_augmented(features)

    def forward_split(self, gamma: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Projections separadas: x_t y gamma no interfieren.
        Mismo diseño que DecompositionEmbedding (trend/seasonal separados).

        x_t → Linear(1, d_model/2)   → señal cruda preservada
        gamma → Linear(K, d_model/2) → info probabilistica de regimen
        concat + LayerNorm → [T, d_model]

        Entrada:
            gamma: Posteriors [T, K] del forward-backward
            x: Valores observados [T] o [T, 1]

        Salida:
            Tensor de embeddings [T, d_model]
        """
        if isinstance(gamma, np.ndarray):
            gamma = torch.from_numpy(gamma).float().to(self.device)
        else:
            gamma = gamma.float().to(self.device)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        else:
            x = x.float().to(self.device)

        if x.dim() > 1:
            x = x.squeeze(-1)

        # Projections separadas
        emb_value = self.projection_value(x.unsqueeze(-1))  # [T, 1] → [T, d_model/2]
        emb_gamma = self.projection_gamma(gamma)             # [T, K] → [T, d_model/2]

        # Concatenar + normalizar
        return self.norm_split(torch.cat([emb_value, emb_gamma], dim=-1))

    def forward_patched(self, gamma: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        HMM + Patching: enriquece serie con gamma y luego parchea.
        Combina lo mejor de HMM (info de regimen) con patching (compresion).

        1. Construye features [x_t, gamma_t] para cada t → [T, 1+K]
        2. Segmenta en patches non-overlapping → [T/P, P*(1+K)]
        3. Proyecta cada patch → [T/P, d_model]

        Entrada:
            gamma: Posteriors [T, K] del forward-backward
            x: Valores observados [T] o [T, 1]

        Salida:
            Tensor de embeddings [T/P, d_model] (6 tokens si T=96, P=16)
        """
        if isinstance(gamma, np.ndarray):
            gamma = torch.from_numpy(gamma).float().to(self.device)
        else:
            gamma = gamma.float().to(self.device)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        else:
            x = x.float().to(self.device)

        if x.dim() > 1:
            x = x.squeeze(-1)

        # [T, 1+K] features por timestep
        features = torch.cat([x.unsqueeze(-1), gamma], dim=-1)  # [T, 1+K]

        T = features.shape[0]
        P = self.patch_len_hmm
        num_patches = T // P

        # Truncar a multiplo de P y reshape a patches
        features = features[:num_patches * P]  # [num_patches*P, 1+K]
        patches = features.view(num_patches, P * features.shape[-1])  # [num_patches, P*(1+K)]

        # Proyectar cada patch
        return self.projection_patched(patches)  # [num_patches, d_model]

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
