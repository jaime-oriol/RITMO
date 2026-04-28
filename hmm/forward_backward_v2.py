"""Forward-Backward HMM v2: JIT compilado con Numba.
Mismo algoritmo y API que forward_backward.py, pero con los bucles compilados
a código nativo (elimina overhead Python de 18k iteraciones secuenciales en T).

Resultado numérico equivalente (logsumexp estable max-trick implementado a mano).
"""

import numpy as np
from typing import Tuple
from numba import njit, prange

from .gaussian_emissions import log_gaussian_emission
from .utils import LOG_EPS, EPS


# ---------------------------------------------------------------------------
# Helpers JIT: logsumexp estable manual, sin scipy
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _lse_pair(a, b):
    """Stable log(exp(a) + exp(b))."""
    if a > b:
        return a + np.log(1.0 + np.exp(b - a))
    return b + np.log(1.0 + np.exp(a - b))


# ---------------------------------------------------------------------------
# Forward batch JIT: bucle T compilado nativo
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True, fastmath=True)
def _forward_batch(log_B, log_A, log_pi):
    """log_B [B,T,K], log_A [K,K], log_pi [K] -> log_alpha [B,T,K], log_lik [B]."""
    B, T, K = log_B.shape
    log_alpha = np.empty((B, T, K))
    log_lik = np.empty(B)

    for b in prange(B):
        # Init t=0
        for k in range(K):
            log_alpha[b, 0, k] = log_pi[k] + log_B[b, 0, k]

        # Forward recursion
        for t in range(1, T):
            for k in range(K):
                # alpha[t,k] = logsumexp_j(alpha[t-1,j] + A[j,k]) + B[t,k]
                mx = log_alpha[b, t-1, 0] + log_A[0, k]
                for j in range(1, K):
                    v = log_alpha[b, t-1, j] + log_A[j, k]
                    if v > mx:
                        mx = v
                s = 0.0
                for j in range(K):
                    s += np.exp(log_alpha[b, t-1, j] + log_A[j, k] - mx)
                log_alpha[b, t, k] = mx + np.log(s) + log_B[b, t, k]

        # Log-likelihood = logsumexp(alpha[T-1, :])
        mx = log_alpha[b, T-1, 0]
        for k in range(1, K):
            if log_alpha[b, T-1, k] > mx:
                mx = log_alpha[b, T-1, k]
        s = 0.0
        for k in range(K):
            s += np.exp(log_alpha[b, T-1, k] - mx)
        log_lik[b] = mx + np.log(s)

    return log_alpha, log_lik


@njit(parallel=True, cache=True, fastmath=True)
def _backward_batch(log_B, log_A):
    """log_B [B,T,K], log_A [K,K] -> log_beta [B,T,K]."""
    B, T, K = log_B.shape
    log_beta = np.zeros((B, T, K))

    for b in prange(B):
        # log_beta[T-1] = 0 ya por np.zeros
        for t in range(T - 2, -1, -1):
            for k in range(K):
                # beta[t,k] = logsumexp_j(A[k,j] + B[t+1,j] + beta[t+1,j])
                mx = log_A[k, 0] + log_B[b, t+1, 0] + log_beta[b, t+1, 0]
                for j in range(1, K):
                    v = log_A[k, j] + log_B[b, t+1, j] + log_beta[b, t+1, j]
                    if v > mx:
                        mx = v
                s = 0.0
                for j in range(K):
                    s += np.exp(log_A[k, j] + log_B[b, t+1, j] + log_beta[b, t+1, j] - mx)
                log_beta[b, t, k] = mx + np.log(s)

    return log_beta


@njit(parallel=True, cache=True, fastmath=True)
def _gamma_xi_stats_batch(log_alpha, log_beta, log_A, log_B, log_lik,
                          obs, need_xi):
    """Calcula gamma agregada y estadísticas suficientes para Baum-Welch.

    Devuelve sums acumulados sobre el batch para evitar materializar gamma/xi
    completos (ahorra memoria masiva con T=18000):
      pi_acc [K]: sum_b gamma[b,0,:]
      A_num [K,K]: sum_b sum_t xi[b,t,:,:]  (si need_xi)
      A_den [K]: sum_b sum_{t<T-1} gamma[b,t,:]
      mu_num [K]: sum_b sum_t gamma[b,t,:] * obs[b,t]
      sq_num [K]: sum_b sum_t gamma[b,t,:] * obs[b,t]^2
      gamma_sum [K]: sum_b sum_t gamma[b,t,:]
    """
    B, T, K = log_alpha.shape

    pi_acc = np.zeros(K)
    A_num = np.zeros((K, K))
    A_den = np.zeros(K)
    mu_num = np.zeros(K)
    sq_num = np.zeros(K)
    gamma_sum = np.zeros(K)

    # Acumuladores thread-locales (evita race condition con prange)
    pi_thr = np.zeros((B, K))
    A_num_thr = np.zeros((B, K, K))
    A_den_thr = np.zeros((B, K))
    mu_num_thr = np.zeros((B, K))
    sq_num_thr = np.zeros((B, K))
    gamma_sum_thr = np.zeros((B, K))

    for b in prange(B):
        ll = log_lik[b]
        # gamma[t,k] = exp(alpha[t,k] + beta[t,k] - ll)
        for t in range(T):
            x = obs[b, t]
            x2 = x * x
            for k in range(K):
                g = np.exp(log_alpha[b, t, k] + log_beta[b, t, k] - ll)
                if t == 0:
                    pi_thr[b, k] = g
                if t < T - 1:
                    A_den_thr[b, k] += g
                mu_num_thr[b, k] += g * x
                sq_num_thr[b, k] += g * x2
                gamma_sum_thr[b, k] += g

        if need_xi:
            for t in range(T - 1):
                for k in range(K):
                    for l in range(K):
                        A_num_thr[b, k, l] += np.exp(
                            log_alpha[b, t, k] + log_A[k, l]
                            + log_B[b, t+1, l] + log_beta[b, t+1, l] - ll
                        )

    # Reducir acumuladores thread-locales
    for b in range(B):
        for k in range(K):
            pi_acc[k] += pi_thr[b, k]
            A_den[k] += A_den_thr[b, k]
            mu_num[k] += mu_num_thr[b, k]
            sq_num[k] += sq_num_thr[b, k]
            gamma_sum[k] += gamma_sum_thr[b, k]
        if need_xi:
            for k in range(K):
                for l in range(K):
                    A_num[k, l] += A_num_thr[b, k, l]

    return pi_acc, A_num, A_den, mu_num, sq_num, gamma_sum


# ---------------------------------------------------------------------------
# API publica equivalente a forward_backward.py
# ---------------------------------------------------------------------------

def forward_backward_batch_v2(observations: np.ndarray,
                              A: np.ndarray,
                              pi: np.ndarray,
                              mu: np.ndarray,
                              sigma: np.ndarray,
                              need_xi: bool = False
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (gamma, log_likelihoods, xi) — gamma y xi materializados (compat con v1).

    Para Baum-Welch en chunks grandes, usar forward_backward_batch_stats_v2 que
    no materializa gamma/xi y es mucho más rápido en memoria.
    """
    if observations.ndim == 1:
        observations = observations[np.newaxis, :]
    if observations.ndim == 3:
        observations = observations.squeeze(-1)

    obs = np.ascontiguousarray(observations, dtype=np.float64)
    B, T = obs.shape
    K = len(mu)

    log_A = np.log(A + LOG_EPS).astype(np.float64)
    log_pi = np.log(pi + LOG_EPS).astype(np.float64)

    # Log emisiones [B, T, K]
    sigma_sq = (sigma.astype(np.float64) ** 2) + EPS
    log_const = -0.5 * np.log(2 * np.pi * sigma_sq)
    diff = obs[:, :, np.newaxis] - mu.astype(np.float64)[np.newaxis, np.newaxis, :]
    log_B = log_const + (-diff ** 2 / (2 * sigma_sq))
    log_B = np.ascontiguousarray(log_B)

    log_alpha, log_lik = _forward_batch(log_B, log_A, log_pi)
    log_beta = _backward_batch(log_B, log_A)

    log_gamma = log_alpha + log_beta - log_lik[:, np.newaxis, np.newaxis]
    gamma = np.exp(log_gamma)

    xi = None
    if need_xi:
        log_xi = (
            log_alpha[:, :-1, :, np.newaxis]
            + log_A[np.newaxis, np.newaxis, :, :]
            + log_B[:, 1:, np.newaxis, :]
            + log_beta[:, 1:, np.newaxis, :]
            - log_lik[:, np.newaxis, np.newaxis, np.newaxis]
        )
        xi = np.exp(log_xi)

    return gamma, log_lik, xi


def forward_backward_batch_stats_v2(observations: np.ndarray,
                                    A: np.ndarray,
                                    pi: np.ndarray,
                                    mu: np.ndarray,
                                    sigma: np.ndarray,
                                    need_xi: bool = True):
    """Devuelve estadísticas agregadas para M-step sin materializar gamma/xi.

    Mucho más rápido y memoria-eficiente para Baum-Welch en chunks grandes.
    Salida: (pi_acc, A_num, A_den, mu_num, sq_num, gamma_sum, total_ll).
    """
    if observations.ndim == 1:
        observations = observations[np.newaxis, :]
    if observations.ndim == 3:
        observations = observations.squeeze(-1)

    obs = np.ascontiguousarray(observations, dtype=np.float64)
    B, T = obs.shape

    log_A = np.log(A + LOG_EPS).astype(np.float64)
    log_pi = np.log(pi + LOG_EPS).astype(np.float64)

    sigma_sq = (sigma.astype(np.float64) ** 2) + EPS
    log_const = -0.5 * np.log(2 * np.pi * sigma_sq)
    diff = obs[:, :, np.newaxis] - mu.astype(np.float64)[np.newaxis, np.newaxis, :]
    log_B = np.ascontiguousarray(log_const + (-diff ** 2 / (2 * sigma_sq)))

    log_alpha, log_lik = _forward_batch(log_B, log_A, log_pi)
    log_beta = _backward_batch(log_B, log_A)

    pi_acc, A_num, A_den, mu_num, sq_num, gamma_sum = _gamma_xi_stats_batch(
        log_alpha, log_beta, log_A, log_B, log_lik, obs, need_xi
    )
    total_ll = log_lik.sum()
    return pi_acc, A_num, A_den, mu_num, sq_num, gamma_sum, total_ll
