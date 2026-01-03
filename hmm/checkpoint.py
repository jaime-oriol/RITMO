"""
Sistema de guardado/carga de parámetros HMM entrenados.

Permite persistir modelos HMM entrenados con Baum-Welch para reutilización
en inferencia (Viterbi) sin re-entrenamiento. Crítico para evaluación zero-shot.
"""

import os
import torch
import numpy as np
from typing import Dict, Any


def save_hmm_params(params: Dict[str, Any], filepath: str) -> None:
    """
    Guarda parámetros HMM entrenados en disco.

    Convierte arrays numpy a tensors PyTorch para compatibilidad con
    el resto del framework (exp/exp_long_term_forecasting usa torch.save).

    Args:
        params: Diccionario retornado por baum_welch() con claves:
            'A': Matriz de transición [K, K]
            'pi': Distribución inicial [K]
            'mu': Medias gaussianas [K]
            'sigma': Desviaciones estándar [K]
            'log_likelihood': Log-verosimilitud final
            'converged': Bool convergencia
            'n_iter': Número de iteraciones
        filepath: Ruta absoluta donde guardar (ej: './cache/hmm_K5.pth')

    Raises:
        ValueError: Si params no contiene claves requeridas
        OSError: Si no se puede crear directorio o escribir archivo

    Ejemplo:
        >>> params = baum_welch(observations, K=5)
        >>> save_hmm_params(params, './cache/hmm_etth1_K5.pth')
    """
    required_keys = {'A', 'pi', 'mu', 'sigma'}
    if not required_keys.issubset(params.keys()):
        missing = required_keys - params.keys()
        raise ValueError(f"params falta claves requeridas: {missing}")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    params_to_save = {
        'A': torch.from_numpy(params['A']).float(),
        'pi': torch.from_numpy(params['pi']).float(),
        'mu': torch.from_numpy(params['mu']).float(),
        'sigma': torch.from_numpy(params['sigma']).float(),
        'log_likelihood': float(params.get('log_likelihood', -np.inf)),
        'converged': bool(params.get('converged', False)),
        'n_iter': int(params.get('n_iter', 0))
    }

    torch.save(params_to_save, filepath)


def load_hmm_params(filepath: str) -> Dict[str, np.ndarray]:
    """
    Carga parámetros HMM desde disco.

    Convierte tensors PyTorch de vuelta a arrays numpy para uso con
    algoritmos HMM (forward_backward, viterbi_decode).

    Args:
        filepath: Ruta absoluta al archivo guardado

    Returns:
        Diccionario con claves:
            'A': np.ndarray [K, K] - Matriz transición
            'pi': np.ndarray [K] - Distribución inicial
            'mu': np.ndarray [K] - Medias gaussianas
            'sigma': np.ndarray [K] - Desviaciones estándar
            'log_likelihood': float
            'converged': bool
            'n_iter': int

    Raises:
        FileNotFoundError: Si filepath no existe
        RuntimeError: Si archivo corrupto o formato inválido

    Ejemplo:
        >>> params = load_hmm_params('./cache/hmm_etth1_K5.pth')
        >>> states, ll = viterbi_decode(obs, params['A'], params['pi'],
        ...                             params['mu'], params['sigma'])
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    params_loaded = torch.load(filepath)

    params = {
        'A': params_loaded['A'].numpy(),
        'pi': params_loaded['pi'].numpy(),
        'mu': params_loaded['mu'].numpy(),
        'sigma': params_loaded['sigma'].numpy(),
        'log_likelihood': params_loaded.get('log_likelihood', -np.inf),
        'converged': params_loaded.get('converged', False),
        'n_iter': params_loaded.get('n_iter', 0)
    }

    return params
