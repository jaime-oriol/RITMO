"""
Técnicas de tokenización para series temporales.

Implementa las 5 categorías principales identificadas en el estado del arte:
1. Discretización - SAX (Lin et al., 2007)
2. Text-based - LLMTime (Gruver et al., 2023)
3. Patching - PatchTST (Nie et al., 2023)
4. Descomposición - Autoformer (Wu et al., 2021)
5. Foundation - MOMENT masked patches (Goswami et al., 2024)

Referencia: Anteproyecto-RITMO.md, Estado del Arte
"""

from .patching import patching_tokenize, visualize_patches
from .decomposition import decomposition_tokenize, visualize_decomposition
from .discretization import sax_discretize, visualize_sax
from .text_based import text_based_tokenize, visualize_text_based
from .foundation import foundation_tokenize, visualize_foundation

__all__ = [
    # Tokenization functions
    'patching_tokenize',
    'decomposition_tokenize',
    'sax_discretize',
    'text_based_tokenize',
    'foundation_tokenize',
    # Visualization functions
    'visualize_patches',
    'visualize_decomposition',
    'visualize_sax',
    'visualize_text_based',
    'visualize_foundation'
]
