"""
Algoritmo Baum-Welch (EM) para entrenamiento de HMM.

Implementa Expectation-Maximization para estimación de máxima verosimilitud
de parámetros λ=(A,π,μ,σ) en Hidden Markov Models (Rabiner 1989).
"""

import numpy as np
from typing import Dict, Any
from tqdm import tqdm
from .forward_backward import forward_backward
from .utils import initialize_kmeans, EPS


def baum_welch(observations: np.ndarray,
               K: int = 5,
               max_iter: int = 100,
               epsilon: float = 1e-4,
               random_state: int = 42,
               verbose: bool = True) -> Dict[str, Any]:
    """
    Entrena HMM mediante algoritmo Baum-Welch (EM).

    Implementa Expectation-Maximization para estimación de parámetros λ=(A,π,μ,σ).
    Garantiza convergencia monótona de log-verosimilitud a máximo local
    (Dempster et al. 1977).

    Algoritmo:
        1. Inicialización: k-means sobre observaciones (Rabiner 1989)
        2. E-step: calcular γ_t(k) y ξ_t(k,l) usando Forward-Backward
        3. M-step: re-estimar parámetros mediante ecuaciones de maximización
        4. Repetir hasta convergencia: |log P(O|λ^n) - log P(O|λ^{n-1})| < ε

    Args:
        observations: Serie temporal normalizada [T]
        K: Número de estados ocultos
        max_iter: Iteraciones máximas EM
        epsilon: Umbral convergencia en log-verosimilitud
        random_state: Semilla para reproducibilidad
        verbose: Mostrar progress bar

    Returns:
        Diccionario con claves:
            'A': Matriz de transición [K, K]
            'pi': Distribución inicial [K]
            'mu': Medias gaussianas [K]
            'sigma': Desviaciones estándar [K]
            'log_likelihood': Log-verosimilitud final
            'converged': True si convergió antes de max_iter
            'n_iter': Número de iteraciones ejecutadas

    Raises:
        ValueError: Si K < 2, observations vacío, o parámetros inválidos

    Ejemplo:
        >>> obs = np.random.randn(1000)  # Serie sintética
        >>> params = baum_welch(obs, K=3, max_iter=50)
        >>> print(f"Convergió: {params['converged']}")
        >>> print(f"Estados: {params['mu']}")
    """
    # ========== 1. VALIDACIONES (Fail Fast) ==========
    if K < 2:
        raise ValueError(f"K debe ser >= 2, recibido: {K}")
    if len(observations) == 0:
        raise ValueError("observations no puede estar vacío")
    if len(observations) < K:
        raise ValueError(f"T={len(observations)} debe ser >= K={K}")
    if max_iter < 1:
        raise ValueError(f"max_iter debe ser >= 1, recibido: {max_iter}")
    if epsilon <= 0:
        raise ValueError(f"epsilon debe ser > 0, recibido: {epsilon}")

    T = len(observations)

    # ========== 2. INICIALIZACIÓN K-MEANS (Rabiner 1989) ==========
    A, pi, mu, sigma = initialize_kmeans(observations, K, random_state)

    # Log-verosimilitud inicial
    log_likelihood_prev = -np.inf
    converged = False

    # ========== 3. LOOP EM ==========
    iterator = range(max_iter)
    if verbose:
        iterator = tqdm(iterator, desc="Baum-Welch EM")

    for iteration in iterator:
        # ===== E-STEP: Calcular estadísticas suficientes =====
        # γ_t(k) = P(q_t=k|O,λ): Probabilidad estado k en timestep t
        # ξ_t(k,l) = P(q_t=k, q_{t+1}=l|O,λ): Probabilidad transición k→l
        gamma, xi, log_likelihood = forward_backward(observations, A, pi, mu, sigma)

        # ===== M-STEP: Re-estimación de parámetros =====

        # π_k = γ_1(k): Probabilidad estado inicial
        pi = gamma[0, :]

        # A_kl = Σ_t ξ_t(k,l) / Σ_t γ_t(k): Transiciones esperadas
        # Numerador: suma de ξ sobre todos los timesteps
        # Denominador: suma de γ sobre timesteps t=0..T-2 (transiciones válidas)
        numerator_A = np.sum(xi, axis=0)  # [K, K]
        denominator_A = np.sum(gamma[:-1, :], axis=0)[:, np.newaxis]  # [K, 1]
        A = numerator_A / (denominator_A + EPS)  # Evitar división por cero

        # μ_k = Σ_t γ_t(k)·o_t / Σ_t γ_t(k): Media ponderada por γ
        numerator_mu = np.sum(gamma * observations[:, np.newaxis], axis=0)  # [K]
        denominator_mu = np.sum(gamma, axis=0)  # [K]
        mu = numerator_mu / (denominator_mu + EPS)

        # σ_k² = Σ_t γ_t(k)·(o_t - μ_k)² / Σ_t γ_t(k): Varianza ponderada
        diff_squared = (observations[:, np.newaxis] - mu)**2  # [T, K]
        numerator_sigma = np.sum(gamma * diff_squared, axis=0)  # [K]
        sigma = np.sqrt(numerator_sigma / (denominator_mu + EPS))

        # Evitar σ=0 que causa problemas numéricos
        sigma = np.maximum(sigma, EPS)

        # ===== CONVERGENCIA: |ΔLL| < ε =====
        delta_ll = abs(log_likelihood - log_likelihood_prev)

        if verbose and iteration % 10 == 0:
            # Actualizar descripción de progress bar cada 10 iteraciones
            iterator.set_postfix({
                'LL': f'{log_likelihood:.2f}',
                'ΔLL': f'{delta_ll:.2e}'
            })

        if delta_ll < epsilon:
            converged = True
            if verbose:
                iterator.close()
                print(f"\nConvergió en iteración {iteration+1}/{max_iter}")
                print(f"  Log-likelihood final: {log_likelihood:.4f}")
            break

        log_likelihood_prev = log_likelihood

    # Si no convergió
    if not converged and verbose:
        print(f"\nNo convergió en {max_iter} iteraciones")
        print(f"  Log-likelihood final: {log_likelihood:.4f}")
        print(f"  ΔLL final: {delta_ll:.2e} (umbral: {epsilon:.2e})")

    # ========== 4. RETORNAR PARÁMETROS ENTRENADOS ==========
    return {
        'A': A,
        'pi': pi,
        'mu': mu,
        'sigma': sigma,
        'log_likelihood': log_likelihood,
        'converged': converged,
        'n_iter': iteration + 1
    }
