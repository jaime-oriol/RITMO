"""
Módulo de capas (layers) compartidas para modelos de series temporales.
Contiene componentes reutilizables:
- Embeddings: codificación de series en vectores (Embed.py)
- Atención: mecanismos de self-attention (SelfAttention_Family.py)
- Encoder/Decoder: bloques de Transformer (Transformer_EncDec.py)
- Descomposición: separar tendencia de estacionalidad (Autoformer_EncDec.py)
- Normalización: RevIN reversible (StandardNorm.py)
"""
