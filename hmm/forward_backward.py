"""
Algoritmo Forward-Backward para HMM.
Calcula la probabilidad de estar en cada estado en cada momento del tiempo.
Es el paso E (Expectation) del algoritmo de entrenamiento.
"""

import numpy as np  # Operaciones matemáticas con arrays
from scipy.special import logsumexp  # Suma de exponenciales estable
from typing import Tuple  # Para indicar tipos de retorno múltiples
from .gaussian_emissions import log_gaussian_emission  # Probabilidades de emisión
from .utils import log_normalize  # Normalización de probabilidades


def _forward(log_B: np.ndarray,
             log_A: np.ndarray,
             log_pi: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Paso FORWARD: Calcula probabilidad de llegar a cada estado desde el inicio.
    Va de izquierda a derecha en el tiempo (t=0 → t=T).

    Pregunta que responde: "¿Cuál es la probabilidad de ver las observaciones
    hasta el tiempo t Y estar en el estado k en ese momento?"

    Entrada:
        log_B: Log-probabilidades de emisión [T, K]
        log_A: Log-probabilidades de transición [K, K]
        log_pi: Log-probabilidades iniciales [K]

    Salida:
        log_alpha: Matriz [T, K] con probabilidades forward
        log_likelihood: Probabilidad total de las observaciones
    """
    T, K = log_B.shape  # T = tiempo, K = número de estados

    # Matriz para guardar resultados: cada celda [t,k] = prob de llegar a estado k en tiempo t
    log_alpha = np.zeros((T, K))

    # PASO 1: Tiempo inicial (t=0)
    # Prob = prob de empezar en estado k × prob de emitir primera observación
    log_alpha[0] = log_pi + log_B[0]

    # PASO 2: Recursión hacia adelante (t=1 hasta T-1)
    for t in range(1, T):
        for k in range(K):
            # Para llegar al estado k en tiempo t:
            # Sumar probabilidades de venir desde TODOS los estados previos j
            # y multiplicar por prob de transición j→k y emisión en k
            log_alpha[t, k] = logsumexp(log_alpha[t-1] + log_A[:, k]) + log_B[t, k]

    # PASO 3: Probabilidad total = suma de probabilidades en todos los estados finales
    log_likelihood = logsumexp(log_alpha[-1])

    return log_alpha, log_likelihood


def _backward(log_B: np.ndarray,
              log_A: np.ndarray) -> np.ndarray:
    """
    Paso BACKWARD: Calcula probabilidad de observaciones futuras desde cada estado.
    Va de derecha a izquierda en el tiempo (t=T → t=0).

    Pregunta que responde: "Si estoy en el estado k en tiempo t,
    ¿cuál es la probabilidad de ver las observaciones restantes?"

    Entrada:
        log_B: Log-probabilidades de emisión [T, K]
        log_A: Log-probabilidades de transición [K, K]

    Salida:
        log_beta: Matriz [T, K] con probabilidades backward
    """
    T, K = log_B.shape  # T = tiempo, K = número de estados

    # Matriz para guardar resultados
    log_beta = np.zeros((T, K))

    # PASO 1: Tiempo final (t=T-1)
    # No hay observaciones futuras, prob = 1, log(1) = 0
    log_beta[-1] = 0.0

    # PASO 2: Recursión hacia atrás (t=T-2 hasta 0)
    for t in range(T-2, -1, -1):
        for k in range(K):
            # Para cada estado k en tiempo t:
            # Sumar probabilidades de ir a TODOS los estados siguientes j
            # considerando: transición k→j, emisión en j, y beta futuro de j
            log_beta[t, k] = logsumexp(log_A[k, :] + log_B[t+1] + log_beta[t+1])

    return log_beta


def _compute_gamma(log_alpha: np.ndarray,
                   log_beta: np.ndarray,
                   log_likelihood: float) -> np.ndarray:
    """
    Calcula GAMMA: probabilidad de estar en cada estado en cada tiempo.
    Combina información del forward (pasado) y backward (futuro).

    Pregunta que responde: "Dadas TODAS las observaciones,
    ¿cuál es la probabilidad de estar en el estado k en el tiempo t?"

    Entrada:
        log_alpha: Probabilidades forward [T, K]
        log_beta: Probabilidades backward [T, K]
        log_likelihood: Probabilidad total de las observaciones

    Salida:
        gamma: Matriz [T, K] con probabilidad de cada estado en cada tiempo
               (valores entre 0 y 1, cada fila suma 1)
    """
    # Combinar forward y backward, normalizar por probabilidad total
    # En log: log(a×b/c) = log(a) + log(b) - log(c)
    log_gamma = log_alpha + log_beta - log_likelihood

    # Convertir de logaritmo a probabilidad normal (exponencial)
    gamma = np.exp(log_gamma)

    return gamma


def _compute_xi(log_alpha: np.ndarray,
                log_beta: np.ndarray,
                log_A: np.ndarray,
                log_B: np.ndarray,
                log_likelihood: float) -> np.ndarray:
    """
    Calcula XI: probabilidad de cada transición entre estados.

    Pregunta que responde: "Dadas TODAS las observaciones,
    ¿cuál es la probabilidad de estar en estado k en tiempo t
    Y pasar al estado l en tiempo t+1?"

    Entrada:
        log_alpha: Probabilidades forward [T, K]
        log_beta: Probabilidades backward [T, K]
        log_A: Log-probabilidades de transición [K, K]
        log_B: Log-probabilidades de emisión [T, K]
        log_likelihood: Probabilidad total de las observaciones

    Salida:
        xi: Tensor [T-1, K, K] donde xi[t,k,l] = prob de transición k→l en tiempo t
    """
    T, K = log_alpha.shape

    # T-1 transiciones (entre t y t+1 para t=0..T-2)
    xi = np.zeros((T-1, K, K))

    # Para cada tiempo t y cada par de estados (k, l)
    for t in range(T-1):
        for k in range(K):  # Estado origen
            for l in range(K):  # Estado destino
                # Prob de transición k→l en tiempo t:
                # = (llegar a k) × (transición k→l) × (emitir en l) × (futuro desde l)
                # Todo dividido por probabilidad total
                log_xi_tkl = (log_alpha[t, k] +      # Prob de llegar a k
                             log_A[k, l] +           # Prob de transición k→l
                             log_B[t+1, l] +         # Prob de emitir observación en l
                             log_beta[t+1, l] -      # Prob del futuro desde l
                             log_likelihood)         # Normalización

                # Convertir de logaritmo a probabilidad normal
                xi[t, k, l] = np.exp(log_xi_tkl)

    return xi


def forward_backward(observations: np.ndarray,
                    A: np.ndarray,
                    pi: np.ndarray,
                    mu: np.ndarray,
                    sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Algoritmo Forward-Backward completo.
    Combina el paso forward y backward para calcular:
    - gamma: probabilidad de cada estado en cada tiempo
    - xi: probabilidad de cada transición entre estados
    - log_likelihood: qué tan bien el modelo explica los datos

    Este es el paso E (Expectation) del algoritmo de entrenamiento.

    Entrada:
        observations: Serie temporal [T valores]
        A: Matriz de transición [K, K] (prob de ir de estado i a j)
        pi: Probabilidades iniciales [K] (prob de empezar en cada estado)
        mu: Medias de cada estado [K]
        sigma: Desviaciones estándar de cada estado [K]

    Salida:
        gamma: [T, K] probabilidad de cada estado en cada tiempo
        xi: [T-1, K, K] probabilidad de cada transición
        log_likelihood: qué tan probable son las observaciones dado el modelo
    """
    K = len(mu)  # Número de estados
    T = len(observations)  # Longitud de la serie

    # Verificar que las dimensiones sean correctas
    if A.shape != (K, K):
        raise ValueError(f"A debe ser [{K},{K}], recibido: {A.shape}")
    if pi.shape != (K,):
        raise ValueError(f"pi debe ser [{K}], recibido: {pi.shape}")

    # PASO 1: Calcular probabilidades de emisión para cada estado y tiempo
    log_B = log_gaussian_emission(observations, mu, sigma)  # [T, K]

    # PASO 2: Convertir todo a escala logarítmica (evita números muy pequeños)
    log_A = np.log(A + 1e-10)  # +1e-10 evita log(0)
    log_pi = np.log(pi + 1e-10)

    # PASO 3: Ejecutar algoritmo forward (del pasado al presente)
    log_alpha, log_likelihood = _forward(log_B, log_A, log_pi)

    # PASO 4: Ejecutar algoritmo backward (del futuro al presente)
    log_beta = _backward(log_B, log_A)

    # PASO 5: Combinar para obtener gamma (prob de cada estado)
    gamma = _compute_gamma(log_alpha, log_beta, log_likelihood)

    # PASO 6: Calcular xi (prob de cada transición)
    xi = _compute_xi(log_alpha, log_beta, log_A, log_B, log_likelihood)

    return gamma, xi, log_likelihood
