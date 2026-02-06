Crea una infografia academica profesional del pipeline RITMO (Regimenes latentes mediante Inferencia Temporal con Markov Oculto) para un Trabajo de Fin de Grado en Computer Science.

Estilo visual: Paper academico de conferencia top-tier (NeurIPS/ICML/ICLR). Fondo blanco limpio, tipografia serif (Computer Modern o similar), paleta sobria con acentos de color para diferenciar etapas. Sin elementos decorativos innecesarios. Diagramas tipo figure de paper con caption formal.

Estructura del pipeline (flujo vertical/horizontal con flechas direccionales):

INPUT — Serie temporal univariada X = [x_1, x_2, ..., x_T], T timesteps, valores continuos en R.

Stage 1: RevIN (Reversible Instance Normalization) — Normalizacion estadistica por instancia. X_norm = (X - mu) / sigma. Se almacenan mu y sigma para desnormalizacion posterior. Referencia: Kim et al., ICLR 2022.

Stage 2: Baum-Welch (HMM Training) — Entrenamiento no supervisado de Hidden Markov Model con K estados ocultos y emisiones gaussianas. Inicializacion via k-means. E-step: Forward-Backward. M-step: Re-estimacion de parametros. Output: lambda* = (A*, B*, pi*). Referencia: Rabiner, 1989; Dempster et al., 1977.

Stage 3: Viterbi (Tokenization) — Decodificacion optima de la secuencia de estados ocultos. Q* = argmax_Q P(Q|O, lambda*). Complejidad O(T*K^2). Output: secuencia de tokens [z_1, z_2, ..., z_T] con z_t in {1, 2, ..., K}. Ratio de compresion ~27x via run-length encoding.

Stage 4: Structured Embeddings — Generacion de vectores interpretables por estado. e_k = [mu_k, sigma_k, A[k,:]]. Dimension: R^(2+K). mu_k captura el centro del regimen, sigma_k la volatilidad, A[k,:] las dinamicas de transicion. Proyeccion lineal a d_model dimensional.

Stage 5: Transformer (Forecasting) — Encoder-only Transformer con pre-norm (LayerNorm). Input: I = 96 embeddings. Multi-head self-attention. Prediccion autoregresiva. y_norm = Transformer(e_{z_1}, ..., e_{z_I}). Horizontes de prediccion O in {96, 192, 336, 720}.

OUTPUT — Desnormalizacion reversible: y_hat = y_norm * sigma + mu. Prediccion final [y_hat_1, y_hat_2, ..., y_hat_O] en escala original.

Elementos adicionales a incluir:
- Flechas con anotaciones matematicas entre etapas
- Codigo de colores consistente: un color distinto por etapa (6 colores, paleta colorblind-friendly Okabe-Ito)
- Notacion matematica formal con LaTeX rendering donde sea posible
- Pequeno recuadro lateral con: "Datasets: ETTh1, ETTh2, Weather, Electricity, Traffic, Exchange" y "Metricas: MSE, MAE"
- Footer con: "RITMO — TFG 2025 | Comparativa vs PatchTST, DLinear, TimeMixer, TimeXer"
- Aspecto ratio horizontal (landscape), resolucion alta, listo para inclusion en documento academico

Tono: Tecnico, preciso, sin adornos. Cada elemento debe aportar informacion. Priorizar claridad y densidad informativa sobre estetica decorativa.
