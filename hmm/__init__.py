"""
Módulo HMM - Hidden Markov Models para tokenización de series temporales.
Convierte series temporales en secuencias de estados discretos (tokens).
"""

# Importa la función de entrenamiento del HMM (algoritmo EM)
from .baum_welch import baum_welch

# Importa funciones para decodificar la secuencia de estados más probable
from .viterbi import viterbi_decode, viterbi_batch

# Importa algoritmo para calcular probabilidades de estados
from .forward_backward import forward_backward

# Importa funciones para guardar/cargar modelos entrenados
from .checkpoint import save_hmm_params, load_hmm_params

# Lista de funciones públicas que se exportan al hacer "from hmm import *"
__all__ = ['baum_welch', 'viterbi_decode', 'viterbi_batch', 'forward_backward',
           'save_hmm_params', 'load_hmm_params']
