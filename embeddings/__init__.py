"""
Módulo de Embeddings para RITMO.

Contiene dos tipos de embeddings:
1. HMM Embedding (EmbeddingGenerator): Embedding interpretable para HMM
   - e_k = [μ_k, σ_k, A[k,:]] con significado físico

2. Technique Embeddings: Embeddings naturales para cada TÉCNICA de tokenización
   (según las 5 técnicas del anteproyecto, NO modelos específicos):
   - DiscretizationEmbedding: Tabla aprendible para técnica de discretización
   - TextBasedEmbedding: Character embeddings para técnica text-based
   - PatchingEmbedding: Proyección lineal para técnica de patching
   - DecompositionEmbedding: Proyección por componente para técnica de descomposición
   - FoundationEmbedding: Patch + mask para técnica de foundation models
"""

# HMM Embedding (propuesta del TFG)
from .embedding_generator import EmbeddingGenerator

# Technique Embeddings (para comparación justa)
# IMPORTANTE: Nombres de TÉCNICAS, no de modelos (ej: "Patching" no "PatchTST")
from .technique_embeddings import (
    DiscretizationEmbedding,
    TextBasedEmbedding,
    PatchingEmbedding,
    DecompositionEmbedding,
    FoundationEmbedding,
    get_embedding,
    EMBEDDINGS,
)

# Lista de clases públicas del módulo
__all__ = [
    # HMM
    'EmbeddingGenerator',
    # Técnicas (5 del anteproyecto)
    'DiscretizationEmbedding',
    'TextBasedEmbedding',
    'PatchingEmbedding',
    'DecompositionEmbedding',
    'FoundationEmbedding',
    # Factory
    'get_embedding',
    'EMBEDDINGS',
]
