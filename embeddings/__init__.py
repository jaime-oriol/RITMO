"""
Módulo de Embeddings para RITMO.
Convierte estados HMM (tokens discretos) en vectores numéricos (embeddings).
Cada embedding contiene: media, desviación y probabilidades de transición del estado.
"""

# Importa la clase principal que genera embeddings
from .embedding_generator import EmbeddingGenerator

# Lista de clases públicas del módulo
__all__ = ['EmbeddingGenerator']
