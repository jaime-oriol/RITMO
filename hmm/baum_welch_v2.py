"""Baum-Welch v2: usa forward_backward_v2 (Numba JIT) y M-step sin materializar gamma/xi.
API compatible con baum_welch.py: misma firma para baum_welch_batch_v2.
"""

import numpy as np
from typing import Dict, Any
from .forward_backward_v2 import forward_backward_batch_stats_v2
from .utils import initialize_kmeans, EPS


def baum_welch_batch_v2(observations: np.ndarray,
                        K: int = 5,
                        max_iter: int = 100,
                        epsilon: float = 1e-4,
                        random_state: int = 42,
                        chunk_size: int = 32,
                        verbose: bool = True) -> Dict[str, Any]:
    """Baum-Welch batch JIT (Numba). Equivalente a baum_welch_batch."""
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim != 2:
        raise ValueError(f"observations debe ser [B, T], recibido: {observations.shape}")
    if K < 2:
        raise ValueError(f"K debe ser >= 2, recibido: {K}")

    B, T = observations.shape
    if T < K:
        raise ValueError(f"T={T} debe ser >= K={K}")

    obs_flat = observations.reshape(-1)
    A, pi, mu, sigma = initialize_kmeans(obs_flat, K, random_state)

    log_likelihood_prev = -np.inf
    log_likelihoods = []
    converged = False

    for iteration in range(max_iter):
        pi_acc = np.zeros(K)
        A_num = np.zeros((K, K))
        A_den = np.zeros(K)
        mu_num = np.zeros(K)
        sq_num = np.zeros(K)
        gamma_sum = np.zeros(K)
        total_ll = 0.0

        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk = observations[start:end]
            (pi_c, A_num_c, A_den_c, mu_num_c, sq_num_c, gamma_sum_c, ll_c
             ) = forward_backward_batch_stats_v2(chunk, A, pi, mu, sigma, need_xi=True)
            total_ll += ll_c
            pi_acc += pi_c
            A_num += A_num_c
            A_den += A_den_c
            mu_num += mu_num_c
            sq_num += sq_num_c
            gamma_sum += gamma_sum_c

        log_likelihoods.append(total_ll)

        # M-step (idéntico a v1)
        pi = pi_acc / B
        pi = pi / (pi.sum() + EPS)

        A = A_num / (A_den[:, np.newaxis] + EPS)
        A = A / A.sum(axis=1, keepdims=True)

        mu_new = mu_num / (gamma_sum + EPS)
        sigma_sq = sq_num / (gamma_sum + EPS) - mu_new ** 2
        sigma = np.sqrt(np.maximum(sigma_sq, EPS ** 2))
        sigma = np.maximum(sigma, EPS)
        mu = mu_new

        if iteration > 0 and total_ll < log_likelihood_prev - 1e-3:
            if verbose:
                print(f"  AVISO: LL_total decreció en iter {iteration+1} "
                      f"({log_likelihood_prev:.4f} → {total_ll:.4f})", flush=True)

        delta_ll = abs(total_ll - log_likelihood_prev)

        if verbose and iteration % 10 == 0:
            print(f"  iter={iteration+1} LL={total_ll:.2f} ΔLL={delta_ll:.2e}", flush=True)

        if iteration > 0 and delta_ll < epsilon:
            converged = True
            if verbose:
                print(f"\nConvergió en iteración {iteration+1}/{max_iter}  LL={total_ll:.4f}", flush=True)
            break

        log_likelihood_prev = total_ll

    if not converged and verbose:
        print(f"\nNo convergió en {max_iter} iteraciones  LL={total_ll:.4f}", flush=True)

    return {
        'A': A,
        'pi': pi,
        'mu': mu,
        'sigma': sigma,
        'log_likelihood': total_ll,
        'log_likelihoods': log_likelihoods,
        'converged': converged,
        'n_iter': iteration + 1,
        'B': B,
        'T': T,
    }
