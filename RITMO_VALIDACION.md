# RITMO Pipeline - Validacion

## Estado: VALIDADO. Listo para modelaje.

## Fase 1: RevIN

Normaliza serie a N(0,1) guardando μ y σ para des-normalizar despues.

**Validacion:**
- MSE reconstruccion: 1.07e-13 (precision numerica)
- Media post-norm: 0.0, Std post-norm: 1.0
- Reversible sin perdida de informacion

## Fase 2: Baum-Welch

Entrena HMM con K=5 estados via algoritmo EM. Aprende parametros A (transiciones), π (inicial), μ (medias), σ (volatilidades).

**Validacion:**
- Convergencia: True (53-200 iters segun dataset)
- Log-likelihood: Mejora monotonica hasta maximo local
- Estados balanceados: 18-21% cada uno, ningun estado muerto

## Fase 3: Viterbi

Tokeniza serie mediante secuencia optima de estados: Q* = argmax P(Q|O,λ)

**Resultados:**
- Vocabulario: K=5 tokens
- Secuencia: T=8640 tokens (1 por timestep)
- Segmentos: 318 (run-length encoding)
- Compresion: 27.17x

**Validacion:**
- Distribucion balanceada (19-21% por token)
- Compresion efectiva (~27 timesteps por segmento)
- Log-prob: -1156.15 (optimo encontrado)

## Fase 4: Embeddings

Convierte cada token k en embedding estructurado: e_k = [μ_k, σ_k, A[k,:]]

**Propiedades:**
- μ_k: Centro del regimen
- σ_k: Volatilidad del regimen
- A[k,:]: Probabilidades de transicion
- Dimension: 7D (2 + K)

**Validacion:**
- Embeddings interpretables (cada dim tiene significado)
- Captura estadisticas (μ, σ) + dinamicas (A)
- Differentiable (compatible con Transformer)

## Checklist Final

1. Normalizacion reversible: MSE < 1e-12
2. HMM converge: Log-likelihood estable, parametros optimos
3. Tokenizacion coherente: Distribucion balanceada, CR=27x
4. Embeddings estructurados: Dim 7D, interpretables

## Siguiente Paso

Crear `models/RITMO.py`:
1. Cargar HMM desde cache
2. Tokenizar batch con viterbi_batch()
3. Embeddings con EmbeddingGenerator
4. Encoder Transformer
5. Proyectar a prediccion
6. Des-normalizar con RevIN
7. Evaluar MSE/MAE vs baselines
