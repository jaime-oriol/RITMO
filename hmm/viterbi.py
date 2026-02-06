"""
Algoritmo de Viterbi para decodificación de HMM.
Encuentra la secuencia de estados MÁS PROBABLE dada una serie de observaciones.
Es como encontrar el mejor camino a través de los estados ocultos.

Referencia: Rabiner (1989), A Tutorial on Hidden Markov Models, IEEE Proc. 77(2), pp. 257-286.
    - Ecuaciones 32a-35 (Viterbi recursion + backtracking)
"""

import numpy as np  # Operaciones matemáticas con arrays
from typing import Tuple  # Para indicar tipos de retorno múltiples
from .gaussian_emissions import log_gaussian_emission  # Probabilidades de emisión
from .utils import LOG_EPS  # Constante para log seguro


def viterbi_decode(observations: np.ndarray,
                   A: np.ndarray,
                   pi: np.ndarray,
                   mu: np.ndarray,
                   sigma: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Encuentra la secuencia de estados más probable para las observaciones dadas.
    Usa programación dinámica para evitar probar todas las combinaciones posibles.

    Idea: En cada momento, solo guardamos el mejor camino para llegar a cada estado.
    Al final, reconstruimos el camino óptimo yendo hacia atrás.

    Entrada:
        observations: Serie temporal [T valores]
        A: Matriz de transición [K, K] (prob de ir de estado i a j)
        pi: Probabilidades iniciales [K]
        mu: Medias de cada estado [K]
        sigma: Desviaciones de cada estado [K]

    Salida:
        state_sequence: Lista de estados [0, 1, 2, ...] para cada tiempo
        log_likelihood: Probabilidad del camino óptimo (qué tan bueno es)
    """
    K = len(mu)  # Número de estados
    T = len(observations)  # Longitud de la serie

    # Verificar dimensiones
    if A.shape != (K, K):
        raise ValueError(f"A debe ser [{K},{K}], recibido: {A.shape}")
    if pi.shape != (K,):
        raise ValueError(f"pi debe ser [{K}], recibido: {pi.shape}")

    # PASO 1: Calcular prob de emisión para cada estado y tiempo
    log_B = log_gaussian_emission(observations, mu, sigma)  # [T, K]

    # PASO 2: Convertir a escala logarítmica
    log_A = np.log(A + LOG_EPS)  # +LOG_EPS evita log(0) = -infinito
    log_pi = np.log(pi + LOG_EPS)  # +LOG_EPS evita log(0) = -infinito

    # PASO 3: Crear tablas para programación dinámica
    delta = np.zeros((T, K))  # delta[t,k] = mejor prob de llegar a estado k en tiempo t
    psi = np.zeros((T, K), dtype=int)  # psi[t,k] = desde qué estado llegamos a k

    # PASO 4: Inicialización (tiempo t=0)
    # Prob = prob inicial × prob de emitir primera observación
    delta[0] = log_pi + log_B[0]

    # PASO 5: Llenar la tabla hacia adelante (t=1 hasta T-1)
    for t in range(1, T):
        for k in range(K):
            # Para cada estado k, buscar el mejor estado previo j
            # scores[j] = prob de estar en j + prob de transición j→k
            scores = delta[t-1] + log_A[:, k]

            # Guardar el mejor estado previo y su probabilidad
            psi[t, k] = np.argmax(scores)  # ¿De dónde venimos?
            delta[t, k] = scores[psi[t, k]] + log_B[t, k]  # Mejor prob + emisión

    # PASO 6: Encontrar el mejor estado final
    last_state = np.argmax(delta[-1])  # Estado con mayor prob en tiempo final
    log_likelihood = delta[-1, last_state]  # Prob del mejor camino

    # PASO 7: Reconstruir el camino óptimo (backtracking)
    state_sequence = np.zeros(T, dtype=int)  # Array para guardar el camino
    state_sequence[-1] = last_state  # Empezamos por el final

    # Ir hacia atrás siguiendo los punteros psi
    for t in range(T-2, -1, -1):
        state_sequence[t] = psi[t+1, state_sequence[t+1]]

    return state_sequence, log_likelihood


def viterbi_batch(observations: np.ndarray,
                  A: np.ndarray,
                  pi: np.ndarray,
                  mu: np.ndarray,
                  sigma: np.ndarray) -> np.ndarray:
    """
    Procesa múltiples series temporales a la vez.
    Aplica viterbi_decode a cada serie del batch de forma independiente.

    Entrada:
        observations: Batch de series [B, T] donde B=número de series, T=longitud
        A: Matriz de transición [K, K]
        pi: Probabilidades iniciales [K]
        mu: Medias de cada estado [K]
        sigma: Desviaciones de cada estado [K]

    Salida:
        state_sequences: Matriz [B, T] con la secuencia de estados de cada serie
    """
    # Si tiene 3 dimensiones [B, T, 1], quitar la última
    if observations.ndim == 3:
        observations = observations.squeeze(-1)

    # Verificar que sea 2D
    if observations.ndim != 2:
        raise ValueError(f"observations debe ser [B,T] o [B,T,1], recibido: {observations.shape}")

    B, T = observations.shape  # B = batch size, T = longitud temporal

    # Verificar que hay al menos una serie
    if B == 0:
        raise ValueError("Batch vacío")

    # Crear matriz de resultados
    state_sequences = np.zeros((B, T), dtype=int)

    # Procesar cada serie del batch
    for b in range(B):
        states, _ = viterbi_decode(observations[b], A, pi, mu, sigma)
        state_sequences[b] = states

    return state_sequences
