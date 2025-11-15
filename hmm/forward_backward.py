"""
Algoritmo Forward-Backward para Hidden Markov Models.

Implementa E-step del algoritmo Baum-Welch mediante cálculo de probabilidades
marginales γ_t(k) y de transición ξ_t(k,l) en log-space (Rabiner 1989).
"""

import numpy as np
from scipy.special import logsumexp
from typing import Tuple
from .gaussian_emissions import log_gaussian_emission
from .utils import log_normalize


def _forward(log_B: np.ndarray,
             log_A: np.ndarray,
             log_pi: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Algoritmo Forward para calcular α_t(k) = P(o_1...o_t, q_t=k|λ).

    Programación dinámica que calcula probabilidad conjunta de observaciones
    hasta timestep t y estado k en timestep t.

    Recursión:
        α_1(k) = π_k × b_k(o_1)
        α_t(k) = [Σ_j α_{t-1}(j) × A_jk] × b_k(o_t)

    Args:
        log_B: Log-emisiones [T, K]
        log_A: Log-transiciones [K, K]
        log_pi: Log-probabilidades iniciales [K]

    Returns:
        Tupla (log_alpha, log_likelihood):
            log_alpha: [T, K] matriz de forward probabilities
            log_likelihood: log P(O|λ)
    """
    T, K = log_B.shape

    # Matriz forward en log-space
    log_alpha = np.zeros((T, K))

    # Inicialización: α_1(k) = π_k × b_k(o_1)
    log_alpha[0] = log_pi + log_B[0]

    # Recursión forward: α_t(k) = [Σ_j α_{t-1}(j) × A_jk] × b_k(o_t)
    for t in range(1, T):
        for k in range(K):
            # Suma sobre todos los estados previos j:
            # log(Σ_j α_{t-1}(j) × A_jk) = logsumexp(log α_{t-1} + log A[:,k])
            log_alpha[t, k] = logsumexp(log_alpha[t-1] + log_A[:, k]) + log_B[t, k]

    # Log-verosimilitud: log P(O|λ) = log Σ_k α_T(k)
    log_likelihood = logsumexp(log_alpha[-1])

    return log_alpha, log_likelihood


def _backward(log_B: np.ndarray,
              log_A: np.ndarray) -> np.ndarray:
    """
    Algoritmo Backward para calcular β_t(k) = P(o_{t+1}...o_T|q_t=k,λ).

    Programación dinámica que calcula probabilidad de observaciones futuras
    dado estado k en timestep t.

    Recursión:
        β_T(k) = 1
        β_t(k) = Σ_j A_kj × b_j(o_{t+1}) × β_{t+1}(j)

    Args:
        log_B: Log-emisiones [T, K]
        log_A: Log-transiciones [K, K]

    Returns:
        log_beta: [T, K] matriz de backward probabilities
    """
    T, K = log_B.shape

    # Matriz backward en log-space
    log_beta = np.zeros((T, K))

    # Inicialización: β_T(k) = 1 → log β_T(k) = 0
    log_beta[-1] = 0.0

    # Recursión backward (de T-1 hacia 1)
    for t in range(T-2, -1, -1):
        for k in range(K):
            # Suma sobre todos los estados siguientes j:
            # log(Σ_j A_kj × b_j(o_{t+1}) × β_{t+1}(j))
            log_beta[t, k] = logsumexp(log_A[k, :] + log_B[t+1] + log_beta[t+1])

    return log_beta


def _compute_gamma(log_alpha: np.ndarray,
                   log_beta: np.ndarray,
                   log_likelihood: float) -> np.ndarray:
    """
    Calcula γ_t(k) = P(q_t=k|O,λ) mediante Forward-Backward.

    Probabilidad marginal de estar en estado k en timestep t dadas
    todas las observaciones O.

    Fórmula:
        γ_t(k) = α_t(k) × β_t(k) / P(O|λ)
        log γ_t(k) = log α_t(k) + log β_t(k) - log P(O|λ)

    Args:
        log_alpha: [T, K] forward probabilities
        log_beta: [T, K] backward probabilities
        log_likelihood: log P(O|λ)

    Returns:
        gamma: [T, K] probabilidades marginales (en espacio normal, no log)
    """
    # Calcular en log-space
    log_gamma = log_alpha + log_beta - log_likelihood

    # Convertir a espacio normal
    gamma = np.exp(log_gamma)

    return gamma


def _compute_xi(log_alpha: np.ndarray,
                log_beta: np.ndarray,
                log_A: np.ndarray,
                log_B: np.ndarray,
                log_likelihood: float) -> np.ndarray:
    """
    Calcula ξ_t(k,l) = P(q_t=k, q_{t+1}=l|O,λ).

    Probabilidad marginal de transición de estado k a estado l entre
    timesteps t y t+1, dadas todas las observaciones O.

    Fórmula:
        ξ_t(k,l) = α_t(k) × A_kl × b_l(o_{t+1}) × β_{t+1}(l) / P(O|λ)

    Args:
        log_alpha: [T, K] forward probabilities
        log_beta: [T, K] backward probabilities
        log_A: [K, K] log-transiciones
        log_B: [T, K] log-emisiones
        log_likelihood: log P(O|λ)

    Returns:
        xi: [T-1, K, K] probabilidades de transición (espacio normal)
    """
    T, K = log_alpha.shape

    # ξ solo se define para t=0..T-2 (transiciones)
    xi = np.zeros((T-1, K, K))

    for t in range(T-1):
        for k in range(K):
            for l in range(K):
                # log ξ_t(k,l) = log α_t(k) + log A_kl + log b_l(o_{t+1}) + log β_{t+1}(l) - log P(O|λ)
                log_xi_tkl = (log_alpha[t, k] +
                             log_A[k, l] +
                             log_B[t+1, l] +
                             log_beta[t+1, l] -
                             log_likelihood)

                # Convertir a espacio normal
                xi[t, k, l] = np.exp(log_xi_tkl)

    return xi


def forward_backward(observations: np.ndarray,
                    A: np.ndarray,
                    pi: np.ndarray,
                    mu: np.ndarray,
                    sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Algoritmo Forward-Backward completo para cálculo de estadísticas suficientes.

    Implementa E-step del algoritmo EM de Baum-Welch (Rabiner 1989).
    Calcula probabilidades marginales γ y transiciones ξ necesarias para
    re-estimación de parámetros en M-step.

    Args:
        observations: Serie temporal [T]
        A: Matriz de transición [K, K]
        pi: Distribución inicial [K]
        mu: Medias gaussianas [K]
        sigma: Desviaciones estándar [K]

    Returns:
        Tupla (gamma, xi, log_likelihood):
            gamma: [T, K] P(q_t=k|O,λ) para cada timestep y estado
            xi: [T-1, K, K] P(q_t=k, q_{t+1}=l|O,λ) para transiciones
            log_likelihood: log P(O|λ) verosimilitud de observaciones

    Raises:
        ValueError: Si dimensiones son inconsistentes
    """
    K = len(mu)
    T = len(observations)

    # Validaciones
    if A.shape != (K, K):
        raise ValueError(f"A debe ser [{K},{K}], recibido: {A.shape}")
    if pi.shape != (K,):
        raise ValueError(f"pi debe ser [{K}], recibido: {pi.shape}")

    # 1. Calcular log-emisiones b_k(o_t) para todos t, k
    log_B = log_gaussian_emission(observations, mu, sigma)  # [T, K]

    # 2. Convertir parámetros a log-space
    log_A = np.log(A + 1e-10)  # Evitar log(0)
    log_pi = np.log(pi + 1e-10)

    # 3. Forward pass: calcular α_t(k)
    log_alpha, log_likelihood = _forward(log_B, log_A, log_pi)

    # 4. Backward pass: calcular β_t(k)
    log_beta = _backward(log_B, log_A)

    # 5. Calcular γ_t(k) = P(q_t=k|O,λ)
    gamma = _compute_gamma(log_alpha, log_beta, log_likelihood)

    # 6. Calcular ξ_t(k,l) = P(q_t=k, q_{t+1}=l|O,λ)
    xi = _compute_xi(log_alpha, log_beta, log_A, log_B, log_likelihood)

    return gamma, xi, log_likelihood
