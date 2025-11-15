"""
Emisiones gaussianas para Hidden Markov Models.

Implementa cálculo de probabilidades de emisión b_k(o_t) = N(o_t; μ_k, σ_k²)
en espacio logarítmico para estabilidad numérica.
"""

import numpy as np
from .utils import EPS


def log_gaussian_emission(observations: np.ndarray,
                          mu: np.ndarray,
                          sigma: np.ndarray) -> np.ndarray:
    """
    Calcula log-probabilidades de emisión gaussiana univariada.

    Implementa log b_k(o_t) = log N(o_t; μ_k, σ_k²) en log-space para evitar
    underflow. Fórmula:
        log N(x; μ, σ²) = -0.5 * log(2π σ²) - (x - μ)² / (2σ²)

    Args:
        observations: Serie temporal [T]
        mu: Medias de los K estados [K]
        sigma: Desviaciones estándar de los K estados [K]

    Returns:
        Log-probabilidades de emisión [T, K] donde entry [t, k] es
        log P(o_t | q_t=k) = log b_k(o_t)

    Raises:
        ValueError: Si sigma contiene valores <= 0

    Ejemplo:
        >>> obs = np.array([1.0, 2.0, 3.0])  # T=3
        >>> mu = np.array([1.5, 3.5])        # K=2
        >>> sigma = np.array([0.5, 1.0])
        >>> log_B = log_gaussian_emission(obs, mu, sigma)
        >>> log_B.shape
        (3, 2)
    """
    # Validación: sigma debe ser positivo
    if np.any(sigma <= 0):
        raise ValueError(f"sigma debe ser > 0, recibido: {sigma}")

    T = len(observations)
    K = len(mu)

    # Broadcasting: [T, 1] y [K] → [T, K]
    obs_expanded = observations[:, np.newaxis]  # [T, 1]
    mu_expanded = mu[np.newaxis, :]            # [1, K]
    sigma_expanded = sigma[np.newaxis, :]      # [1, K]

    # Añadir EPS para estabilidad numérica (evita log(0) o división por cero)
    sigma_squared = sigma_expanded**2 + EPS

    # Término constante: -0.5 * log(2π σ²)
    log_const = -0.5 * np.log(2 * np.pi * sigma_squared)

    # Término exponencial: -(x - μ)² / (2σ²)
    diff_squared = (obs_expanded - mu_expanded)**2
    log_exp = -diff_squared / (2 * sigma_squared)

    # Log-probabilidad gaussiana
    log_B = log_const + log_exp  # [T, K]

    return log_B
