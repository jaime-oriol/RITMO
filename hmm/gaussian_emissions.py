"""
Emisiones gaussianas para HMM.
Calcula qué tan probable es que cada estado genere cada observación.
Usa distribución gaussiana (campana de Gauss) para modelar las emisiones.

Referencia: Rabiner (1989), A Tutorial on Hidden Markov Models, IEEE Proc. 77(2), pp. 257-286.
"""

import numpy as np  # Librería para operaciones matemáticas
from .utils import EPS  # Constante pequeña para evitar errores numéricos


def log_gaussian_emission(observations: np.ndarray,
                          mu: np.ndarray,
                          sigma: np.ndarray) -> np.ndarray:
    """
    Calcula la probabilidad de que cada estado emita cada observación.
    Usa la fórmula de la distribución normal (gaussiana).

    Entrada:
        observations: Serie temporal con T valores
        mu: Media (centro) de cada uno de los K estados
        sigma: Desviación estándar (dispersión) de cada estado

    Salida:
        Matriz [T, K] donde cada celda [t, k] dice:
        "¿Qué tan probable es que el estado k genere el valor observado en tiempo t?"
        (En escala logarítmica para evitar números muy pequeños)
    """
    # Verificar que sigma sea positivo (no tiene sentido dispersión negativa)
    if np.any(sigma <= 0):
        raise ValueError(f"sigma debe ser > 0, recibido: {sigma}")

    T = len(observations)  # Número de puntos en el tiempo
    K = len(mu)  # Número de estados

    # Expandir dimensiones para poder operar matrices de diferentes tamaños
    # observations: [T] → [T, 1] (columna)
    # mu y sigma: [K] → [1, K] (fila)
    # Resultado de operaciones: [T, K] (matriz completa)
    obs_expanded = observations[:, np.newaxis]  # Cada observación en una fila
    mu_expanded = mu[np.newaxis, :]  # Cada media en una columna
    sigma_expanded = sigma[np.newaxis, :]  # Cada sigma en una columna

    # Varianza = sigma al cuadrado (+ EPS para evitar división por cero)
    sigma_squared = sigma_expanded**2 + EPS

    # Fórmula gaussiana en logaritmo (dos partes):
    # Parte 1: Constante de normalización
    log_const = -0.5 * np.log(2 * np.pi * sigma_squared)

    # Parte 2: Qué tan lejos está la observación de la media
    diff_squared = (obs_expanded - mu_expanded)**2  # Distancia al cuadrado
    log_exp = -diff_squared / (2 * sigma_squared)  # Penalización por distancia

    # Suma de ambas partes = log-probabilidad gaussiana
    log_B = log_const + log_exp  # Matriz [T, K]

    return log_B
