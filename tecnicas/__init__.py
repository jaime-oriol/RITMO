"""
Técnicas de tokenización para series temporales.

Implementa las 5 categorías principales identificadas en el estado del arte:
1. Discretización - Cuantización uniforme/SAX/VQ-VAE
2. Text-based - Conversión a strings numéricos
3. Patching - Segmentación en ventanas fijas
4. Descomposición - Trend + Seasonal + Residual
5. Foundation - Masked patches estilo MOMENT

Referencia: Anteproyecto-RITMO.md, Estado del Arte
"""

from .patching import patching_tokenize
from .decomposition import decomposition_tokenize
from .discretization import discretization_tokenize
from .text_based import text_based_tokenize
from .foundation import foundation_tokenize

__all__ = [
    'patching_tokenize',
    'decomposition_tokenize',
    'discretization_tokenize',
    'text_based_tokenize',
    'foundation_tokenize'
]
