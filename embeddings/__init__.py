"""
Módulo de Embeddings Estructurados para RITMO.

Transforma tokens discretos (estados HMM) en embeddings vectoriales
e_k = [μ_k, σ_k, A[k,:]] que encapsulan información estadística y
dinámica temporal.
"""

from .embedding_generator import EmbeddingGenerator

__all__ = ['EmbeddingGenerator']
