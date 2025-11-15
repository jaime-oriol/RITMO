"""
Utilidades para algoritmos HMM.

Proporciona funciones helper para normalización en log-space e inicialización
de parámetros HMM mediante k-means.
"""

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from typing import Tuple

# Constantes para estabilidad numérica
EPS = 1e-5  # Evita divisiones por cero
LOG_ZERO = -1e10  # Representa log(0) de forma segura


def log_normalize(log_probs: np.ndarray) -> np.ndarray:
    """
    Normaliza probabilidades en log-space.

    Calcula log(p_i / Σ p_j) = log(p_i) - log(Σ p_j) de forma numéricamente
    estable usando logsumexp. Evita underflow al trabajar directamente en
    espacio logarítmico.

    Args:
        log_probs: Array de log-probabilidades

    Returns:
        Log-probabilidades normalizadas que suman 1 en espacio normal

    Ejemplo:
        >>> log_p = np.array([-1.0, -2.0, -3.0])
        >>> log_norm = log_normalize(log_p)
        >>> np.exp(log_norm).sum()  # ≈ 1.0
    """
    # logsumexp(x) calcula log(Σ exp(x_i)) sin underflow
    # Normalización: log(p_i) - log(Σ p_j)
    return log_probs - logsumexp(log_probs)


def initialize_kmeans(observations: np.ndarray,
                      K: int,
                      random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inicializa parámetros HMM mediante k-means clustering.

    Estrategia estándar de Rabiner (1989):
    - μ_k: Centroides de k-means
    - σ_k: Desviación estándar global (conservadora)
    - π_k: Frecuencias de clusters
    - A: Matriz uniforme con ruido pequeño

    Args:
        observations: Serie temporal [T]
        K: Número de estados ocultos
        random_state: Semilla para reproducibilidad

    Returns:
        Tupla (A, pi, mu, sigma):
            A: Matriz de transición [K, K]
            pi: Distribución inicial [K]
            mu: Medias gaussianas [K]
            sigma: Desviaciones estándar [K]

    Raises:
        ValueError: Si K > len(observations)
    """
    if K > len(observations):
        raise ValueError(f"K={K} no puede ser mayor que T={len(observations)}")

    # Reshape para k-means: [T] → [T, 1]
    obs_reshaped = observations.reshape(-1, 1)

    # Clustering k-means
    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(obs_reshaped)

    # μ_k: Centroides como medias iniciales
    mu = kmeans.cluster_centers_.flatten()

    # σ_k: Desviación estándar global para todos los estados (conservador)
    # Evita clusters con varianza cero
    global_std = np.std(observations)
    sigma = np.full(K, max(global_std, EPS))

    # π_k: Frecuencia de cada cluster como probabilidad inicial
    pi = np.bincount(labels, minlength=K).astype(float)
    pi = pi / pi.sum()  # Normalizar

    # A: Matriz de transición uniforme con ruido pequeño
    # Ruido evita probabilidades exactamente 0 que pueden causar problemas
    A = np.ones((K, K)) / K
    noise = np.random.RandomState(random_state).uniform(-0.01, 0.01, (K, K))
    A = A + noise
    # Normalizar filas para que sumen 1
    A = A / A.sum(axis=1, keepdims=True)

    return A, pi, mu, sigma
