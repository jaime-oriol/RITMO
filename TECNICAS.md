# Tecnicas de Tokenizacion

## 1. HMM (Viterbi) - RITMO

**Tipo:** Probabilistico (Baum-Welch + Viterbi)

**Tokenizacion:**
- Vocabulario: K=5 estados
- Secuencia: T tokens (1 por timestep)
- Segmentos: Run-length encoding de cambios
- Compresion: 27x (ETTh1)

**Embeddings:** e_k = [μ_k, σ_k, A[k,:]] - Estructurados, interpretables

**Trade-off:** Requiere entrenamiento HMM, asume gaussianas

## 2. SAX (Lin et al., 2007)

**Tipo:** Discretizacion con breakpoints gaussianos

**Tokenizacion:**
- Vocabulario: 8 simbolos {a,b,c,...,h}
- Secuencia: T tokens (1 por timestep)
- Compresion: 1x (sin compresion)

**Trade-off:** Simple pero pierde informacion, sin compresion

## 3. LLMTime (Gruver et al., 2023)

**Tipo:** Text-based (conversion a strings numericos)

**Tokenizacion:**
- Vocabulario: 13 chars (digitos + signo + punto + sep)
- Secuencia: ~10T caracteres
- Compresion: 0.1x (expansion)

**Trade-off:** Compatible con LLMs pre-entrenados pero expansion de datos

## 4. PatchTST (Nie et al., 2023)

**Tipo:** Patch-based (segmentacion en ventanas)

**Tokenizacion:**
- Vocabulario: Continuo (patches ∈ R^16)
- Secuencia: T/16 patches
- Compresion: 16x

**Trade-off:** Reduce complejidad Transformer pero patches arbitrarios sin significado estadistico

## 5. Autoformer (Wu et al., 2021)

**Tipo:** Decomposition (trend + seasonal)

**Tokenizacion:**
- Vocabulario: 2 componentes
- Secuencia: 2 tokens (cada uno longitud T)
- Compresion: 750x

**Trade-off:** Separacion frecuencias clara pero solo 2 tokens (muy comprimido)

## 6. MOMENT (Goswami et al., 2024)

**Tipo:** Foundation model (masked patch reconstruction)

**Tokenizacion:**
- Vocabulario: Continuo (embeddings learnables)
- Secuencia: T/16 patches (30% masked)
- Compresion: 16x

**Trade-off:** Pre-training self-supervised pero requiere fase costosa de entrenamiento

## Comparativa

| Tecnica | Vocab | Tokens (T=1500) | Segmentos | CR | Interpretable |
|---------|-------|-----------------|-----------|----|----|
| HMM | K=5 | 1500 | 318 | 27x | Alta (μ,σ,A) |
| SAX | 8 | 1500 | 1500 | 1x | Alta (simbolos) |
| LLMTime | 13 | 15000 | N/A | 0.1x | Baja (chars) |
| PatchTST | Cont. | 93 | 93 | 16x | Media (ventanas) |
| Autoformer | 2 | 2 | 2 | 750x | Alta (trend+seasonal) |
| MOMENT | Cont. | 93 | 93 | 16x | Baja (black box) |

## Implementacion

**Codigo:** `tecnicas/` (discretization.py, text_based.py, patching.py, decomposition.py, foundation.py)

**HMM:** `hmm/` (baum_welch.py, viterbi.py)

**Notebooks:** `tecnicas/ETTh2_tokenization.ipynb`, `tecnicas/Electricity_tokenization.ipynb`
