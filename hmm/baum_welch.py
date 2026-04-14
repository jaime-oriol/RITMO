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
from .forward_backward import forward_backward, forward_backward_batch  # Paso E del algoritmo
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
    # 1. Verificar entradas
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

    # 2. Inicializacion con k-means
    # Crear parámetros iniciales agrupando datos con k-means
    A, pi, mu, sigma = initialize_kmeans(observations, K, random_state)

    # Variables para seguimiento del entrenamiento
    log_likelihood_prev = -np.inf  # Empezamos con "infinitamente malo"
    log_likelihoods = []  # Guardamos historial para ver convergencia
    converged = False  # Aún no ha convergido

    # 3. Bucle principal EM
    # Preparar iterador (con o sin barra de progreso)
    iterator = range(max_iter)
    if verbose:
        iterator = tqdm(iterator, desc="Baum-Welch EM")

    for iteration in iterator:
        # Paso E (Expectation): estimar estados
        # gamma[t,k] = probabilidad de estar en estado k en tiempo t
        # xi[t,k,l] = probabilidad de transición k→l en tiempo t
        gamma, xi, log_likelihood = forward_backward(observations, A, pi, mu, sigma)
        log_likelihoods.append(log_likelihood)  # Guardar para historial

        # Paso M (Maximization): actualizar parametros

        # Actualizar pi: probabilidad inicial = prob del primer estado
        pi = gamma[0, :]

        # Actualizar A: matriz de transición
        # = (veces esperadas de transición k→l) / (veces esperadas en estado k)
        numerator_A = np.sum(xi, axis=0)  # Suma de transiciones k→l
        denominator_A = np.sum(gamma[:-1, :], axis=0)[:, np.newaxis]  # Veces en k
        A = numerator_A / (denominator_A + EPS)  # División segura
        # Re-normalizar filas para garantizar propiedad estocástica (sum=1)
        A = A / A.sum(axis=1, keepdims=True)

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

        # Verificar convergencia
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

        # Si ya no mejora significativamente, parar (saltar iter 0 donde prev=-inf)
        if iteration > 0 and delta_ll < epsilon:
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

    # 4. Devolver resultados
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


def baum_welch_batch(observations: np.ndarray,
                     K: int = 5,
                     max_iter: int = 100,
                     epsilon: float = 1e-4,
                     random_state: int = 42,
                     chunk_size: int = 32,
                     verbose: bool = True) -> Dict[str, Any]:
    """
    Baum-Welch para multiples secuencias independientes (channel-independence).

    Entrena un unico conjunto de parametros (A, pi, mu, sigma) agregando estadisticas
    suficientes sobre todas las secuencias del batch. Uso tipico en Plan B:
    apilar los C canales de un dataset como B secuencias y entrenar un HMM compartido.

    Implementacion memory-aware: procesa el batch en chunks para evitar que xi
    [B, T-1, K, K] explote la RAM (e.g. ECL con C=321, T~18000, K=10 = 4.6GB).
    Usa identidad sigma^2 = E[x^2] - mu^2 para un solo paso E por iteracion.

    Entrada:
        observations: [B, T] batch de B secuencias de longitud T
        K: numero de estados ocultos
        max_iter: maximo de iteraciones EM
        epsilon: umbral de convergencia |delta_LL_total|
        random_state: semilla para k-means
        chunk_size: tamanio del mini-batch interno para memoria acotada
        verbose: barra de progreso

    Salida: mismo diccionario que baum_welch() pero 'log_likelihood' es la suma
        sobre las B secuencias.
    """
    observations = np.asarray(observations)
    if observations.ndim != 2:
        raise ValueError(f"observations debe ser [B, T], recibido: {observations.shape}")
    if K < 2:
        raise ValueError(f"K debe ser >= 2, recibido: {K}")

    B, T = observations.shape
    if T < K:
        raise ValueError(f"T={T} debe ser >= K={K}")

    # Inicializacion k-means sobre la concatenacion de todas las secuencias
    # para capturar la distribucion global de valores a traves de canales.
    obs_flat = observations.reshape(-1)
    A, pi, mu, sigma = initialize_kmeans(obs_flat, K, random_state)

    log_likelihood_prev = -np.inf
    log_likelihoods = []
    converged = False

    iterator = range(max_iter)
    if verbose:
        iterator = tqdm(iterator, desc="Baum-Welch batch EM")

    for iteration in iterator:
        # Acumuladores de estadisticas suficientes agregadas sobre el batch
        pi_acc = np.zeros(K)                # sum_b gamma[b, 0, :]
        A_num = np.zeros((K, K))            # sum_b sum_t xi[b, t, :, :]
        A_den = np.zeros(K)                 # sum_b sum_{t<T-1} gamma[b, t, :]
        mu_num = np.zeros(K)                # sum_b sum_t gamma[b, t, :] * x[b, t]
        sq_num = np.zeros(K)                # sum_b sum_t gamma[b, t, :] * x[b, t]^2
        gamma_sum = np.zeros(K)             # sum_b sum_t gamma[b, t, :]
        total_ll = 0.0

        # Procesar el batch en chunks para mantener xi acotado en memoria
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk = observations[start:end]
            gamma, ll_chunk, xi = forward_backward_batch(
                chunk, A, pi, mu, sigma, need_xi=True
            )  # gamma: [b, T, K], xi: [b, T-1, K, K]

            total_ll += ll_chunk.sum()
            pi_acc += gamma[:, 0, :].sum(axis=0)
            A_num += xi.sum(axis=(0, 1))
            A_den += gamma[:, :-1, :].sum(axis=(0, 1))

            # Estadisticas para mu y sigma en un solo paso (via E[x^2])
            mu_num += (gamma * chunk[:, :, np.newaxis]).sum(axis=(0, 1))
            sq_num += (gamma * (chunk[:, :, np.newaxis] ** 2)).sum(axis=(0, 1))
            gamma_sum += gamma.sum(axis=(0, 1))

        log_likelihoods.append(total_ll)

        # M-step: actualizar parametros con las estadisticas agregadas
        pi = pi_acc / B
        pi = pi / (pi.sum() + EPS)

        A = A_num / (A_den[:, np.newaxis] + EPS)
        A = A / A.sum(axis=1, keepdims=True)

        mu_new = mu_num / (gamma_sum + EPS)
        # sigma^2 = E[x^2] - (E[x])^2, con clipping para estabilidad
        sigma_sq = sq_num / (gamma_sum + EPS) - mu_new ** 2
        sigma = np.sqrt(np.maximum(sigma_sq, EPS ** 2))
        sigma = np.maximum(sigma, EPS)
        mu = mu_new

        # Monotonicidad esperada (EM) con tolerancia numerica
        if iteration > 0 and total_ll < log_likelihood_prev - 1e-3:
            if verbose:
                print(f"  AVISO: LL_total decreció en iter {iteration+1} "
                      f"({log_likelihood_prev:.4f} → {total_ll:.4f})")

        delta_ll = abs(total_ll - log_likelihood_prev)

        if verbose and iteration % 10 == 0:
            iterator.set_postfix({'LL_total': f'{total_ll:.2f}', 'ΔLL': f'{delta_ll:.2e}'})

        if iteration > 0 and delta_ll < epsilon:
            converged = True
            if verbose:
                iterator.close()
                print(f"\nConvergió en iteración {iteration+1}/{max_iter}")
                print(f"  Log-likelihood total: {total_ll:.4f}")
            break

        log_likelihood_prev = total_ll

    if not converged and verbose:
        print(f"\nNo convergió en {max_iter} iteraciones")
        print(f"  Log-likelihood total final: {total_ll:.4f}")
        print(f"  ΔLL final: {delta_ll:.2e} (umbral: {epsilon:.2e})")

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
