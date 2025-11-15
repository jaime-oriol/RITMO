"""
Algoritmo de Viterbi para decodificación de HMM.

Encuentra la secuencia de estados ocultos más probable Q* = argmax P(Q|O,λ)
mediante programación dinámica (Rabiner 1989).
"""

import numpy as np
from typing import Tuple
from .gaussian_emissions import log_gaussian_emission


def viterbi_decode(observations: np.ndarray,
                   A: np.ndarray,
                   pi: np.ndarray,
                   mu: np.ndarray,
                   sigma: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Decodifica secuencia óptima de estados mediante algoritmo de Viterbi.

    Encuentra Q* = argmax_Q P(Q|O,λ) usando programación dinámica con
    complejidad O(T×K²), evitando exploración exhaustiva O(K^T).

    Variables:
        δ_t(k): Máxima log-probabilidad de camino hasta t terminando en k
        ψ_t(k): Estado previo óptimo en timestep t-1 para backtracking

    Recursión:
        δ_1(k) = log π_k + log b_k(o_1)
        δ_t(k) = max_j [δ_{t-1}(j) + log A_jk] + log b_k(o_t)
        ψ_t(k) = argmax_j [δ_{t-1}(j) + log A_jk]

    Args:
        observations: Serie temporal [T]
        A: Matriz de transición [K, K]
        pi: Distribución inicial [K]
        mu: Medias gaussianas [K]
        sigma: Desviaciones estándar [K]

    Returns:
        Tupla (state_sequence, log_likelihood):
            state_sequence: [T] array de índices de estados (0..K-1)
            log_likelihood: log P(Q*, O|λ) del camino óptimo

    Raises:
        ValueError: Si dimensiones son inconsistentes

    Ejemplo:
        >>> obs = np.array([1.0, 5.2, 5.1, 1.1, 0.9])
        >>> A = np.array([[0.7, 0.3], [0.4, 0.6]])
        >>> pi = np.array([0.5, 0.5])
        >>> mu = np.array([1.0, 5.0])
        >>> sigma = np.array([0.5, 0.5])
        >>> states, ll = viterbi_decode(obs, A, pi, mu, sigma)
        >>> states  # [0, 1, 1, 0, 0] por ejemplo
    """
    K = len(mu)
    T = len(observations)

    # Validaciones
    if A.shape != (K, K):
        raise ValueError(f"A debe ser [{K},{K}], recibido: {A.shape}")
    if pi.shape != (K,):
        raise ValueError(f"pi debe ser [{K}], recibido: {pi.shape}")

    # 1. Calcular log-emisiones b_k(o_t)
    log_B = log_gaussian_emission(observations, mu, sigma)  # [T, K]

    # 2. Convertir parámetros a log-space (evitar log(0))
    log_A = np.log(A + 1e-10)
    log_pi = np.log(pi + 1e-10)

    # 3. Inicializar variables de programación dinámica
    delta = np.zeros((T, K))     # δ_t(k): máxima log-prob hasta t
    psi = np.zeros((T, K), dtype=int)  # ψ_t(k): estado previo óptimo

    # 4. Inicialización: δ_1(k) = log π_k + log b_k(o_1)
    delta[0] = log_pi + log_B[0]
    # psi[0] no se usa (no hay estado previo)

    # 5. Recursión: δ_t(k) = max_j [δ_{t-1}(j) + log A_jk] + log b_k(o_t)
    for t in range(1, T):
        for k in range(K):
            # Calcular δ_{t-1}(j) + log A_jk para todos los j
            scores = delta[t-1] + log_A[:, k]

            # Maximización
            psi[t, k] = np.argmax(scores)  # Estado previo óptimo
            delta[t, k] = scores[psi[t, k]] + log_B[t, k]  # Máxima log-prob

    # 6. Terminación: encontrar estado final con máxima probabilidad
    last_state = np.argmax(delta[-1])
    log_likelihood = delta[-1, last_state]

    # 7. Backtracking: recuperar secuencia óptima
    state_sequence = np.zeros(T, dtype=int)
    state_sequence[-1] = last_state

    # Recorrer hacia atrás usando punteros ψ
    for t in range(T-2, -1, -1):
        state_sequence[t] = psi[t+1, state_sequence[t+1]]

    return state_sequence, log_likelihood
