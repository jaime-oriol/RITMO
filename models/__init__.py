"""
Modelos baseline para predicción de series temporales.
Contiene 4 arquitecturas de referencia para comparar con RITMO (HMM):
- DLinear: Descomposición + capas lineales (simple pero efectivo)
- PatchTST: Transformer con patches (tokenización por segmentos)
- TimeMixer: Mezcla multi-escala con descomposición
- TimeXer: Transformer con variables exógenas
- TransformerCommon: Transformer encoder-only para Plan A (comparación controlada)
"""
