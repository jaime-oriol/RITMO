"""
Utilidades para algoritmos HMM.
Funciones auxiliares: normalización y inicialización de parámetros.

Referencia: Rabiner (1989), A Tutorial on Hidden Markov Models, IEEE Proc. 77(2), pp. 257-286.
"""

import numpy as np  # Librería para operaciones matemáticas con arrays
from scipy.special import logsumexp  # Suma de exponenciales estable numéricamente
from sklearn.cluster import KMeans  # Algoritmo de clustering para inicialización
from typing import Tuple  # Para indicar tipos de retorno múltiples

# Constantes de seguridad
EPS = 1e-5  # Número pequeño para evitar dividir por cero en denominadores
LOG_EPS = 1e-10  # Número pequeño para evitar log(0) = -infinito
LOG_ZERO = -1e10  # Representa log(0) sin causar errores


def log_normalize(log_probs: np.ndarray) -> np.ndarray:
    """
    Normaliza probabilidades para que sumen 1.
    Trabaja en espacio logarítmico para evitar números muy pequeños.

    Entrada: array de log-probabilidades (pueden no sumar 1)
    Salida: array de log-probabilidades normalizadas (suman 1 en escala normal)
    """
    # Resta el total para que las probabilidades sumen 1
    # logsumexp calcula log(suma de exponenciales) de forma segura
    return log_probs - logsumexp(log_probs)


def initialize_kmeans(observations: np.ndarray,
                      K: int,
                      random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Crea valores iniciales para el HMM usando k-means.
    K-means agrupa los datos en K grupos, y usamos esos grupos como punto de partida.

    Entrada:
        observations: Serie temporal (lista de valores)
        K: Número de estados/grupos que queremos
        random_state: Semilla para resultados reproducibles

    Salida: 4 arrays (A, pi, mu, sigma) que son los parámetros iniciales del HMM
    """
    # Verificar que hay suficientes datos para K estados
    if K > len(observations):
        raise ValueError(f"K={K} no puede ser mayor que T={len(observations)}")

    # K-means necesita datos en forma de columna: [T] → [T, 1]
    obs_reshaped = observations.reshape(-1, 1)

    # Ejecutar k-means: agrupa los datos en K clusters
    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(obs_reshaped)  # Asigna cada punto a un cluster

    # mu: Centro de cada cluster (valor típico de cada estado)
    mu = kmeans.cluster_centers_.flatten()

    # sigma: Dispersión de cada cluster (adaptada, no global uniforme)
    global_std = np.std(observations)  # Desviación estándar global como fallback
    sigma = np.empty(K)  # Array para guardar dispersión de cada estado
    for k in range(K):
        mask = (labels == k)  # Seleccionar puntos asignados al cluster k
        if mask.sum() > 1:  # Si el cluster tiene más de 1 punto
            sigma[k] = max(np.std(observations[mask]), EPS)  # Std real del cluster
        else:
            sigma[k] = max(global_std, EPS)  # Fallback: std global si solo 1 punto

    # pi: Probabilidad de empezar en cada estado (según frecuencia de clusters)
    pi = np.bincount(labels, minlength=K).astype(float)  # Cuenta puntos por cluster
    pi = pi / pi.sum()  # Normaliza para que sume 1

    # A: Matriz de transición (probabilidad de ir de estado i a estado j)
    # Empieza uniforme (todos igual de probables) con ruido pequeño
    A = np.ones((K, K)) / K  # Matriz K×K con valor 1/K en cada celda
    noise = np.random.RandomState(random_state).uniform(-0.01, 0.01, (K, K))
    A = A + noise  # Añade pequeña variación aleatoria
    A = A / A.sum(axis=1, keepdims=True)  # Normaliza cada fila para que sume 1

    return A, pi, mu, sigma
