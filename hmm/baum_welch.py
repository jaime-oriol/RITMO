"""
Algoritmo Baum-Welch (EM) para entrenamiento de HMM.
Aprende los parámetros óptimos del modelo a partir de los datos.
EM = Expectation-Maximization: alterna entre estimar estados y actualizar parámetros.

Referencias:
    - Rabiner (1989), A Tutorial on Hidden Markov Models, IEEE Proc. 77(2), pp. 257-286.
    - Dempster et al. (1977), Maximum Likelihood from Incomplete Data via the EM Algorithm,
      JRSS Series B, 39(1), pp. 1-38.
"""

import numpy as np  # Operaciones matemáticas con arrays
from typing import Dict, Any  # Para indicar tipos de datos
from tqdm import tqdm  # Barra de progreso visual
from .forward_backward import forward_backward  # Paso E del algoritmo
from .utils import initialize_kmeans, EPS  # Inicialización y constante de seguridad


def baum_welch(observations: np.ndarray,
               K: int = 5,
               max_iter: int = 100,
               epsilon: float = 1e-4,
               random_state: int = 42,
               verbose: bool = True) -> Dict[str, Any]:
    """
    Entrena un HMM usando el algoritmo Baum-Welch.
    Encuentra los mejores parámetros (A, pi, mu, sigma) para explicar los datos.

    El algoritmo alterna entre:
    - Paso E: Estimar en qué estado estaba el sistema en cada momento
    - Paso M: Actualizar parámetros basándose en esas estimaciones

    Entrada:
        observations: Serie temporal (lista de valores numéricos)
        K: Número de estados ocultos que queremos descubrir
        max_iter: Máximo de vueltas del algoritmo (por si no converge)
        epsilon: Cuándo parar (si mejora menos que esto, terminamos)
        random_state: Semilla para resultados reproducibles
        verbose: Si True, muestra barra de progreso

    Salida: Diccionario con:
        'A': Matriz de transición [K, K] - probabilidad de ir de estado i a j
        'pi': Probabilidades iniciales [K] - prob de empezar en cada estado
        'mu': Medias [K] - valor típico de cada estado
        'sigma': Desviaciones [K] - dispersión de cada estado
        'log_likelihood': Qué tan bien el modelo explica los datos
        'converged': True si terminó porque ya no mejoraba
        'n_iter': Cuántas iteraciones hizo
    """
    # ========== 1. VERIFICAR ENTRADAS ==========
    # Comprobar que los parámetros tienen sentido
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

    T = len(observations)  # Longitud de la serie temporal

    # ========== 2. INICIALIZACIÓN CON K-MEANS ==========
    # Crear parámetros iniciales agrupando datos con k-means
    A, pi, mu, sigma = initialize_kmeans(observations, K, random_state)

    # Variables para seguimiento del entrenamiento
    log_likelihood_prev = -np.inf  # Empezamos con "infinitamente malo"
    log_likelihoods = []  # Guardamos historial para ver convergencia
    converged = False  # Aún no ha convergido

    # ========== 3. BUCLE PRINCIPAL EM ==========
    # Preparar iterador (con o sin barra de progreso)
    iterator = range(max_iter)
    if verbose:
        iterator = tqdm(iterator, desc="Baum-Welch EM")

    for iteration in iterator:
        # ===== PASO E (Expectation): Estimar estados =====
        # gamma[t,k] = probabilidad de estar en estado k en tiempo t
        # xi[t,k,l] = probabilidad de transición k→l en tiempo t
        gamma, xi, log_likelihood = forward_backward(observations, A, pi, mu, sigma)
        log_likelihoods.append(log_likelihood)  # Guardar para historial

        # ===== PASO M (Maximization): Actualizar parámetros =====

        # Actualizar pi: probabilidad inicial = prob del primer estado
        pi = gamma[0, :]

        # Actualizar A: matriz de transición
        # = (veces esperadas de transición k→l) / (veces esperadas en estado k)
        numerator_A = np.sum(xi, axis=0)  # Suma de transiciones k→l
        denominator_A = np.sum(gamma[:-1, :], axis=0)[:, np.newaxis]  # Veces en k
        A = numerator_A / (denominator_A + EPS)  # División segura

        # Actualizar mu: media de cada estado
        # = promedio de observaciones ponderado por prob de estar en ese estado
        numerator_mu = np.sum(gamma * observations[:, np.newaxis], axis=0)
        denominator_mu = np.sum(gamma, axis=0)
        mu = numerator_mu / (denominator_mu + EPS)

        # Actualizar sigma: desviación de cada estado
        # = dispersión de observaciones ponderada por prob de estar en ese estado
        diff_squared = (observations[:, np.newaxis] - mu)**2  # Distancia al cuadrado
        numerator_sigma = np.sum(gamma * diff_squared, axis=0)
        sigma = np.sqrt(numerator_sigma / (denominator_mu + EPS))
        sigma = np.maximum(sigma, EPS)  # Evitar sigma=0

        # ===== VERIFICAR CONVERGENCIA =====
        # Monotonicidad: EM garantiza que LL no decrece (Dempster 1977)
        if iteration > 0 and log_likelihood < log_likelihood_prev - 1e-6:
            if verbose:
                print(f"  AVISO: LL decreció en iter {iteration+1} "
                      f"({log_likelihood_prev:.4f} → {log_likelihood:.4f}), posible inestabilidad numérica")

        # Si la mejora es menor que epsilon, terminamos
        delta_ll = abs(log_likelihood - log_likelihood_prev)

        # Mostrar progreso cada 10 iteraciones
        if verbose and iteration % 10 == 0:
            iterator.set_postfix({
                'LL': f'{log_likelihood:.2f}',
                'ΔLL': f'{delta_ll:.2e}'
            })

        # Si ya no mejora significativamente, parar
        if delta_ll < epsilon:
            converged = True
            if verbose:
                iterator.close()
                print(f"\nConvergió en iteración {iteration+1}/{max_iter}")
                print(f"  Log-likelihood final: {log_likelihood:.4f}")
            break

        # Guardar para comparar en siguiente iteración
        log_likelihood_prev = log_likelihood

    # Mensaje si no convergió
    if not converged and verbose:
        print(f"\nNo convergió en {max_iter} iteraciones")
        print(f"  Log-likelihood final: {log_likelihood:.4f}")
        print(f"  ΔLL final: {delta_ll:.2e} (umbral: {epsilon:.2e})")

    # ========== 4. DEVOLVER RESULTADOS ==========
    # Empaquetar todos los parámetros entrenados en un diccionario
    return {
        'A': A,                          # Matriz de transición aprendida
        'pi': pi,                        # Probabilidades iniciales aprendidas
        'mu': mu,                        # Medias de cada estado
        'sigma': sigma,                  # Desviaciones de cada estado
        'log_likelihood': log_likelihood,  # Calidad final del modelo
        'log_likelihoods': log_likelihoods,  # Historial de mejora
        'converged': converged,          # ¿Terminó por convergencia?
        'n_iter': iteration + 1          # Número de iteraciones realizadas
    }
