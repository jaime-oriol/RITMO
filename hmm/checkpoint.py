"""
Sistema de guardado/carga de parámetros HMM entrenados.
Permite guardar un modelo entrenado y cargarlo después sin re-entrenar.
Útil para no perder el trabajo de entrenamiento.
"""

import os  # Operaciones con archivos y carpetas
import torch  # Librería de deep learning (usamos su formato de guardado)
import numpy as np  # Operaciones matemáticas
from typing import Dict, Any  # Para indicar tipos de datos


def save_hmm_params(params: Dict[str, Any], filepath: str) -> None:
    """
    Guarda los parámetros del HMM entrenado en un archivo.

    Entrada:
        params: Diccionario con los parámetros del modelo (de baum_welch):
            'A': Matriz de transición
            'pi': Probabilidades iniciales
            'mu': Medias de cada estado
            'sigma': Desviaciones de cada estado
        filepath: Ruta donde guardar (ej: './cache/hmm_etth1_K5.pth')
    """
    # Verificar que tenemos todos los parámetros necesarios
    required_keys = {'A', 'pi', 'mu', 'sigma'}
    if not required_keys.issubset(params.keys()):
        missing = required_keys - params.keys()
        raise ValueError(f"params falta claves requeridas: {missing}")

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convertir arrays numpy a tensores PyTorch (formato estándar de guardado)
    params_to_save = {
        'A': torch.from_numpy(params['A']).float(),      # Matriz transición
        'pi': torch.from_numpy(params['pi']).float(),    # Probs iniciales
        'mu': torch.from_numpy(params['mu']).float(),    # Medias
        'sigma': torch.from_numpy(params['sigma']).float(),  # Desviaciones
        'log_likelihood': float(params.get('log_likelihood', -np.inf)),  # Calidad
        'converged': bool(params.get('converged', False)),  # ¿Convergió?
        'n_iter': int(params.get('n_iter', 0))  # Iteraciones usadas
    }

    # Guardar en disco
    torch.save(params_to_save, filepath)


def load_hmm_params(filepath: str) -> Dict[str, np.ndarray]:
    """
    Carga parámetros HMM guardados previamente.

    Entrada:
        filepath: Ruta al archivo guardado (ej: './cache/hmm_etth1_K5.pth')

    Salida:
        Diccionario con los parámetros del modelo:
            'A': Matriz de transición (numpy array)
            'pi': Probabilidades iniciales
            'mu': Medias de cada estado
            'sigma': Desviaciones de cada estado
            'log_likelihood': Calidad del modelo
            'converged': Si convergió durante entrenamiento
            'n_iter': Iteraciones que tomó entrenar
    """
    # Verificar que el archivo existe
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    # Cargar desde disco
    params_loaded = torch.load(filepath)

    # Convertir tensores PyTorch de vuelta a arrays numpy
    params = {
        'A': params_loaded['A'].numpy(),      # Matriz transición
        'pi': params_loaded['pi'].numpy(),    # Probs iniciales
        'mu': params_loaded['mu'].numpy(),    # Medias
        'sigma': params_loaded['sigma'].numpy(),  # Desviaciones
        'log_likelihood': params_loaded.get('log_likelihood', -np.inf),
        'converged': params_loaded.get('converged', False),
        'n_iter': params_loaded.get('n_iter', 0)
    }

    return params
