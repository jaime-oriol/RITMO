"""
Módulo HMM - Hidden Markov Models para tokenización de series temporales.

Implementa algoritmo Baum-Welch (EM) para entrenamiento de HMM con emisiones
gaussianas, Forward-Backward para inferencia, y Viterbi para decodificación.

Referencias:
    - Rabiner (1989): A Tutorial on Hidden Markov Models
    - Dempster et al. (1977): Maximum Likelihood from Incomplete Data via EM
"""

from .baum_welch import baum_welch
from .viterbi import viterbi_decode, viterbi_batch
from .forward_backward import forward_backward
from .checkpoint import save_hmm_params, load_hmm_params

__all__ = ['baum_welch', 'viterbi_decode', 'viterbi_batch', 'forward_backward',
           'save_hmm_params', 'load_hmm_params']
