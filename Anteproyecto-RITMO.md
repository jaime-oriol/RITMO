# MEMORIA JUSTIFICATIVA PARA LA ACEPTACIÓN DEL ANTEPROYECTO

**Título:** Tokenizar series temporales con técnicas existentes y contrastar resultados empleando estados ocultos de Markov

**Autor:** Jaime Oriol Goicoechea

## Acrónimo del proyecto:

**RITMO** - **R**egímenes latentes mediante **I**nferencia **T**emporal con **M**arkov **O**culto para tokenización y forecasting de series temporales

---

## DESCRIPCIÓN DEL PROBLEMA A RESOLVER

### ✓ Motivación y origen.

La aplicación de Large Language Models (LLMs) a series temporales ha experimentado un desarrollo considerable en años recientes, motivado por el éxito de arquitecturas transformer en procesamiento de lenguaje natural. No obstante, la transferencia de estas arquitecturas al dominio temporal enfrenta un desafío fundamental: las series temporales son inherentemente continuas, mientras que los LLMs operan sobre secuencias discretas de tokens. Esta disparidad requiere una transformación previa que convierta observaciones numéricas en representaciones discretas procesables, proceso denominado tokenización, que constituye un cuello de botella crítico en el pipeline completo.

Los métodos existentes se clasifican en cinco categorías principales. La discretización (Lin et al., 2007; van den Oord et al., 2017; Ansari et al., 2024; Talukder et al., 2024) transforma valores continuos en símbolos discretos mediante cuantización determinística o aprendizaje de codebooks. Los enfoques text-based (Gruver et al., 2023; Jin et al., 2024) convierten observaciones numéricas en cadenas procesables por tokenizadores lingüísticos. El patching (Nie et al., 2023; Zhang & Aggarwal, 2023; Peršak et al., 2024; Abeywickrama et al., 2025) segmenta series en subsecuencias que actúan como tokens, con variantes desde ventanas fijas hasta boundaries adaptativos determinados por entropía condicional. La descomposición (Wu et al., 2021; Zhou et al., 2022; Cao et al., 2024; Woo et al., 2022) separa series en componentes aditivos (trend/seasonal/residual) modelados independientemente. Los modelos fundacionales (Goswami et al., 2024; Liu et al., 2024; Woo et al., 2024; Zhou et al., 2023) aprenden representaciones universales mediante pre-entrenamiento masivo sobre millones de series heterogéneas.

Pese a la diversidad de enfoques, cada categoría presenta limitaciones inherentes. La discretización ofrece granularidad fija sin adaptación a variabilidad intrínseca de la serie. Los enfoques text-based dependen de representaciones implícitas aprendidas de corpus textuales sin garantías sobre captura de propiedades temporales. El patching con ventanas de longitud constante impone segmentación arbitraria que puede fragmentar patrones coherentes. La descomposición mantiene dependencias entre componentes de forma implícita. Los modelos fundacionales aprenden dependencias mediante atención sobre millones de ejemplos sin estructura probabilística subyacente.

En contraposición, los Hidden Markov Models (HMM) ofrecen estructura probabilística que modela explícitamente transiciones entre estados ocultos mediante cadena de Markov de primer orden. Cada estado k representa un régimen estadístico con parámetros de emisión propios (media μ_k, varianza σ²_k), y las transiciones se rigen por matriz de probabilidad A donde A_ij = P(z_t = j | z_{t-1} = i). Esta característica permite que estados ocultos actúen como embeddings latentes que encapsulan simultáneamente información estadística local y estructura temporal explícita, proporcionando representación más rica que métodos determinísticos actuales.

### ✓ Datos que lo sustentan.

Wang et al. (2025) establecen relación exponencial entre complejidad de patrones temporales y error mínimo alcanzable: MSE ≈ exp(α·Complexity), demostrando que múltiples benchmarks (MSE, MAE) en series ampliamente utilizadas (ETT, Weather, Electricity) han alcanzado saturación. Este hallazgo cuantifica la necesidad de explorar enfoques con estructura probabilística explícita.

Los surveys recientes proporcionan el marco conceptual. Abdullahi et al. (2025) realizan revisión sistemática identificando tokenización efectiva como desafío fundamental abierto. Zhang et al. (2024) documentan limitaciones en preservación de dependencias para horizontes largos (>336 timesteps). Jiang et al. (2024) identifican gaps en captura de dependencias temporales complejas mediante atención implícita. Liang et al. (2024) establecen paradigma de foundation models, permitiendo contrastar embeddings implícitos aprendidos mediante masked reconstruction frente a embeddings estructurados con significado estadístico explícito. Wen et al. (2023) documentan arquitecturas transformer baseline, Informer (Zhou et al., 2021) con ProbSparse attention, TimesNet (Wu et al., 2023) con transformación 2D-variation, y DLinear (Zeng et al., 2023) demostrando que descomposición lineal simple supera a transformers complejos, proporcionando contexto arquitectónico para integración de tokenización HMM.

La efectividad de HMM se sustenta en evidencia empírica. Hamilton (1989) demostró que modelos de Markov-Switching superan a ARIMA lineales en series económicas con múltiples regímenes estadísticos. Yeh & Tang (2022) reportan ganancias de +8.2 puntos NMI y +3.7 puntos F1 empleando Neural HMMs con dependencias markovianas explícitas frente a VQ-APC sin estructura temporal para segmentación de audio. Tang & Matteson (2021) logran resultados SOTA en forecasting mediante ProTran, combinando State-Space Models con atención transformer, validando que estados ocultos con dinámica temporal explícita actúan efectivamente como embeddings intermedios.

### ✓ Impacto de la solución del problema.

La integración de HMM como mecanismo de tokenización probabilística permitiría avanzar en tres direcciones críticas. Primero, generaría embeddings estructurados e_k = [μ_k, σ_k, A[k,:]] donde cada dimensión posee significado estadístico interpretable: μ_k caracteriza valor central del régimen, σ_k cuantifica volatilidad, y A[k,:] ∈ ℝ^K codifica explícitamente dinámica temporal mediante P(z_t = j | z_{t-1} = k). Esta representación contrasta con embeddings implícitos en foundation models donde dimensiones latentes carecen de interpretación directa.

Segundo, proporcionaría marco teórico riguroso basado en algoritmo EM para entrenamiento (Dempster et al., 1977) garantizando convergencia monótona a máximo local, y programación dinámica para inferencia, forward-backward (Rabiner, 1989) y Viterbi, con complejidad O(T·K²) produciendo soluciones exactas. Esta fundamentación contrasta con métodos heurísticos de discretización sin justificación teórica formal.

Tercero, establecería alternativa con estructura probabilística explícita que podría superar límite de saturación identificado por Wang et al. (2025), especialmente en series con cambios de régimen pronunciados donde métodos sin memoria explícita fallan en capturar transiciones entre dinámicas estadísticamente distintas.

### ✓ Problema de investigación en forma de Pregunta.

¿Pueden los estados ocultos de un Hidden Markov Model actuar como embeddings latentes estructurados que capturen dependencias temporales y regímenes estadísticos de manera más efectiva que las técnicas determinísticas actuales de tokenización para la predicción de series temporales univariadas en el contexto de modelos de lenguaje?

### ✓ Preguntas de investigación (Research Questions, RQ)

**RQ1:** ¿Cuáles son las técnicas de tokenización y embedding actuales para series temporales en el contexto de Large Language Models, y cuáles son sus limitaciones específicas en la captura de dependencias temporales, adaptabilidad a cambios de régimen y preservación de información estructurada?

**RQ2:** ¿Qué propiedades matemáticas y estadísticas de los Hidden Markov Models los hacen adecuados para la representación latente de series temporales con cambios de régimen, y cómo se han aplicado históricamente estos modelos a la segmentación temporal y modelización de dependencias markovianas?

**RQ3:** ¿Cómo pueden los estados ocultos de un Hidden Markov Model reinterpretarse como mecanismo de tokenización donde cada estado representa un embedding vectorial estructurado e_k = [μ_k, σ_k, A[k,:]] que encapsula simultáneamente información estadística local y dinámica temporal global, y cómo se compara teóricamente esta representación frente a métodos determinísticos?

**RQ4:** ¿Qué métricas de evaluación cuantitativas son apropiadas para validar la calidad de tokenización basada en HMM en preservación de información, captura de dependencias y desempeño downstream en predicción univariada?

**RQ5:** ¿Cuáles son los trade-offs cuantitativos entre complejidad computacional (tiempo de entrenamiento con Baum-Welch, inferencia con Viterbi), ratio de compresión y desempeño en forecasting univariado al emplear tokenización HMM versus métodos determinísticos?

### ✓ Estado del arte.

#### Contexto general: LLMs y series temporales

La aplicación de arquitecturas transformer al dominio temporal ha experimentado desarrollo acelerado en años recientes. Los surveys recientes sistematizan este campo emergente desde múltiples perspectivas. Abdullahi et al. (2025) realizan revisión sistemática de más de 100 trabajos identificando tres paradigmas principales: adaptación de LLMs pre-entrenados mediante fine-tuning, arquitecturas diseñadas específicamente para series temporales, y métodos de tokenización que transforman observaciones continuas en secuencias discretas. Zhang et al. (2024) documentan aplicaciones LLM específicamente a forecasting, clasificación y detección de anomalías, identificando limitaciones en preservación de dependencias para horizontes largos. Liang et al. (2024) establecen taxonomía que distingue entre modelos generativos, discriminativos y representacionales, permitiendo posicionar estados ocultos HMM como embeddings con estructura probabilística explícita. Wen et al. (2023) documentan arquitecturas transformer baseline consolidadas, proporcionando contexto arquitectónico necesario. Jiang et al. (2024) identifican explícitamente limitaciones en captura de dependencias complejas mediante atención estándar, motivando búsqueda de alternativas con prior estructurado.

#### Técnicas de tokenización actuales

##### Discretización

Lin et al. (2007) proponen Symbolic Aggregate approXimation (SAX), técnica pionera que reduce dimensionalidad mediante PAA y discretización alfabética, garantizando lower-bound de distancia Euclidiana. Sin embargo, presenta granularidad fija y ausencia de memoria temporal explícita. van den Oord et al. (2017) introducen Vector Quantized Variational AutoEncoders (VQ-VAE), que aprenden codebooks discretos mediante entrenamiento end-to-end con straight-through estimator para gradientes. Talukder et al. (2024) desarrollan TOTEM, aplicando VQ-VAE con tokenización exclusivamente temporal generando vocabulario de ~256 tokens agnóstico al dominio, logrando ratio de compresión 8:1 manteniendo capacidad representacional. Ansari et al. (2024) desarrollan Chronos, estado del arte actual que tokeniza mediante mean scaling y cuantización uniforme en ~4096 tokens, entrenando T5/GPT-2 sin modificaciones arquitecturales e incorporando data augmentation mediante TSMixup y KernelSynth para compensar escasez de datos. Zhao et al. (2024) proponen Sparse-VQ, que aplica vector quantization después del encoder para discretizar embeddings y reducir ruido, eliminando módulo FFN del transformer porque RevIN + VQ capturan estadísticas necesarias.

##### Patching

Nie et al. (2023) introducen PatchTST, que segmenta series en patches como tokens de entrada reduciendo complejidad de atención de O(L²) a O((L/S)²) e implementando channel-independence procesando cada canal con pesos compartidos. Proponen masked autoencoding a nivel de patches para pre-entrenamiento self-supervised. Abeywickrama et al. (2025) desarrollan EntroPE con patching dinámico donde boundaries se determinan mediante entropía condicional calculada por transformer pre-entrenado, colocándolos en puntos de alta incertidumbre predictiva que marcan transiciones naturales. Peršak et al. (2024) proponen patching multi-resolución (MRP) dividiendo series en patches a múltiples escalas simultáneamente (K={1,2,4,8,16}) alimentando todos los tokens resultantes a un único mecanismo de atención para modelar explícitamente relaciones cross-scale. Zhang & Aggarwal (2023) desarrollan Crossformer con tokenización DSW (Dimension-Segment-Wise) que embebe segmentos de cada dimensión separadamente generando array 2D, implementando atención Two-Stage (TSA) que modela dependencias temporales y entre variables mediante routers. Zhang et al. (2024) implementan arquitectura multi-rama donde cada rama tokeniza con tamaño de patch diferente (P₁=8, P₂=32, P₃=96) capturando simultáneamente patrones de alta frecuencia y tendencias de largo plazo.

##### Descomposición

Wu et al. (2021) proponen Autoformer, que descompone series mediante módulos de autocorrelación progresiva extrayendo tendencia mediante promedio móvil y modelando estacionalidad con atención. Zhou et al. (2022) desarrollan FEDformer aplicando transformada de Fourier para descomposición en dominio frecuencial, implementando attention sparse en modo Fourier para captura eficiente de periodicidades. Cao et al. (2024) desarrollan TEMPO, que adapta GPT-2 pre-entrenado mediante descomposición STL obligatoria (trend/seasonal/residual) con prompts semi-soft por componente inicializados con embeddings textuales entrenables, demostrando zero-shot cross-domain entrenando en datasets heterogéneos. Woo et al. (2022) proponen CoST, que aprende representaciones desacopladas seasonal-trend mediante contrastive learning con data augmentation temporal y temporal shift, maximizando acuerdo entre vistas de un mismo componente y minimizando acuerdo entre componentes distintos.

##### Foundation Models

Goswami et al. (2024) desarrollan MOMENT, familia de foundation models basados en T5 architecture pre-entrenados sobre Time Series Pile (compilación masiva de datasets públicos), implementando masked reconstruction con 30% de patches enmascarados. Proponen multiple input resolutions y multi-dataset training con vocabulario compartido. Liu et al. (2024) proponen Timer, que pre-entrena GPT-2 sobre datos agregados de múltiples dominios mediante single-step autoregressive forecasting, implementando tokenización mediante discretización cuantílica adaptativa que preserva información de magnitud. Woo et al. (2024) desarrollan MOIRAI, unified forecasting transformer pre-entrenado mediante mixture of distributions approach que modela simultáneamente diferentes tipos de series (univariadas, multivariadas, irregulares), implementando any-variate attention mechanism. Zhou et al. (2023) proponen FPT (One Fits All), que pre-entrena modelo universal mediante frozen large language model como encoder de secuencias temporales convertidas a texto, demostrando transferencia efectiva de conocimiento lingüístico a dominio temporal.

##### Text-based

Gruver et al. (2023) proponen LLMTime, enfoque zero-shot que convierte series temporales en strings numéricos procesados directamente por LLMs como GPT-3/GPT-4 mediante next-token prediction, demostrando competitividad con métodos especializados sin entrenamiento específico. Jin et al. (2024) desarrollan Time-LLM, que reprograma LLMs mediante text prototypes aprendidos que alinean embeddings temporales con espacio lingüístico, permitiendo aprovechar capacidades de reasoning pre-entrenadas manteniendo LLM frozen mediante adaptadores.

##### Transformers Eficientes y Baselines Consolidados

Zhou et al. (2021) proponen Informer, que introduce ProbSparse self-attention reduciendo complejidad de O(L²logL) a O(LlogL) mediante selección de queries dominantes, implementando self-attention distilling para manejo eficiente de secuencias largas. Wu et al. (2023) desarrollan TimesNet, que transforma series 1D en tensores 2D mediante reshaping basado en periodicidades detectadas, aplicando convoluciones 2D para captura simultánea de variaciones intra-period y inter-period. Zeng et al. (2023) proponen DLinear, baseline simple que descompone mediante promedio móvil y aplica dos capas lineales separadas a trend y seasonal, demostrando sorprendentemente que este enfoque simple supera a transformers complejos en múltiples benchmarks cuestionando necesidad de arquitecturas sofisticadas. Wang et al. (2024b) desarrollan TimeMixer, que implementa descomposición multiscale mixing aplicando mixing operations separadamente a diferentes escalas temporales (past-decomposable-mixing para patrones históricos, future-multipredictor-mixing para horizontes distintos), logrando modelado eficiente de dependencias multi-resolución. Wang et al. (2024a) proponen TimeXer, transformer que incorpora explícitamente variables exógenas conocidas mediante módulo de exogenous-aware attention que modula atención endógena con información externa, demostrando ganancias significativas en forecasting cuando covariates relevantes están disponibles.

##### Preprocesamiento y Normalización

Kim et al. (2022) introducen Reversible Instance Normalization (RevIN), que normaliza cada instancia independientemente almacenando estadísticas (μ, σ) para desnormalización posterior, mitigando distribution shift entre train y test. Demuestran mejoras consistentes al aplicar RevIN como wrapper genérico a arquitecturas existentes. Liu et al. (2022) proponen Non-stationary Transformers con Series Stationarization (normalización bidireccional preservando estadísticas originales) y De-stationary Attention (reincorporando información no-estacionaria mediante factores aprendidos que rescalan atención), evitando over-stationarization donde normalización genera atenciones indistinguibles. Jibao et al. (2025) desarrollan Inner-instance Normalization, técnica que normaliza dentro de cada instancia mediante ventanas deslizantes adaptativas capturando estadísticas locales time-varying, demostrando robustez superior frente a non-stationarity extrema.

#### Hidden Markov Models

##### Fundamentos Teóricos

Dempster et al. (1977) presentan algoritmo EM (Expectation-Maximization) para estimación de máxima verosimilitud con datos incompletos, alternando entre E-step (calcular esperanza de estadísticas suficientes) y M-step (maximizar verosimilitud). Demuestran convergencia monótona e incluyen aplicación explícita a HMM con estados latentes markovianos. Rabiner (1989) sistematiza teoría de HMMs mediante tutorial que presenta algoritmos fundamentales: forward-backward para evaluación de P(O|λ), Viterbi para decodificación óptima de estados mediante programación dinámica con complejidad O(T·K²), y Baum-Welch para aprendizaje de parámetros λ=(A,B,π) mediante EM.

##### Regime-Switching y Estructura Temporal

Hamilton (1989) introduce modelo de Markov-Switching donde parámetros de autoregresión cambian según estado latente discreto que evoluciona como cadena de Markov, desarrollando algoritmo de filtrado análogo a filtro de Kalman para inferir probabilísticamente estados ocultos. Aplicado al PIB de EE.UU., demuestra que ciclo económico se caracteriza mejor mediante cambios discretos entre regímenes que mediante modelos ARIMA lineales. Tang & Matteson (2021) desarrollan ProTran, que combina State-Space Models con arquitecturas Transformer usando atención para modelar dinámicas no-Markovianas en espacio latente, eliminando RNNs. Implementa capas jerárquicas de variables latentes estocásticas con inferencia variacional para forecasting probabilístico no-autoregresivo generando predicciones con incertidumbre mediante transiciones estado-a-estado.

##### Selección Automática de Estados

Fox et al. (2011) proponen sticky HDP-HMM que añade parámetro κ de auto-transición al HDP-HMM estándar aumentando probabilidad previa de permanecer en mismo estado mediante πj|α,κ,β ∼ DP(α+κ, (αβ+κδj)/(α+κ)), controlando persistencia temporal y evitando sobre-segmentación. Permite emisiones no paramétricas mediante DP mixtures of Gaussians y desarrolla blocked Gibbs sampler sobre aproximación truncada del DP.

##### Aplicaciones Modernas

Dai et al. (2017) proponen R-HSMM (Recurrent Hidden Semi-Markov Model), que extiende HSMM clásico reemplazando asunciones paramétricas simples de emisión por RNNs generativas donde cada estado oculto tiene RNN propia modelando secuencia de observaciones dentro de cada segmento. Desarrollan método de penalización distribucional estocástica que entrena simultáneamente modelo generativo y bi-RNN encoder aproximando forward-backward. Mensch & Blondel (2018) proponen framework para hacer diferenciables algoritmos de programación dinámica mediante suavizado del operador max usando regularizadores fuertemente convexos (negentropía o norma L2), presentando DPΩ(θ) como operador convexo diferenciable cuyo gradiente corresponde a trayectoria esperada de random walk permitiendo backpropagation eficiente, con aplicación a Viterbi suavizado y DTW suavizado. Yeh & Tang (2022) desarrollan Neural HMMs que modelan dependencias Markovianas entre tokens discretos de audio mediante transition distributions p(zt|zt-1, x1:t-k) parametrizadas por productos externos de representaciones de estados, introduciendo low frame-rate HMMs con "hops" para capturar slowness temporal, entrenados end-to-end con forward-backward algorithm.

#### Gap identificado y oportunidad de investigación

A pesar de los avances en tokenización para series temporales y la efectividad demostrada de HMM en modelización de dependencias temporales, no existe en la literatura una propuesta que emplee sistemáticamente estados ocultos de HMM como mecanismo de tokenización para LLMs aplicados a series temporales. Las técnicas actuales presentan carencias complementarias: discretización y patching carecen de estructura probabilística explícita, text-based y foundation models producen embeddings sin interpretabilidad directa, y descomposición no modela explícitamente transiciones entre regímenes. Los HMM ofrecen simultáneamente estructura probabilística rigurosa fundamentada en teoría estadística consolidada (algoritmo EM, programación dinámica), embeddings interpretables donde cada estado representa un régimen con parámetros estadísticos explícitos (μ_k, σ_k), y captura de dinámica temporal mediante matriz de transición A que codifica dependencias markovianas. Esta combinación única posiciona a los HMM como alternativa fundamentada para avanzar más allá de las limitaciones de métodos determinísticos actuales.

#### Benchmarks, métricas y el problema de saturación

La comunidad científica ha consolidado benchmarks estandarizados para evaluación sistemática mediante Time-Series-Library (TSLib), repositorio de código abierto que proporciona implementaciones unificadas y protocolos experimentales consistentes. Los datasets principales incluyen ETT (Electricity Transformer Temperature) en cuatro variantes, Electricity, Traffic, Weather, Exchange, ILI y M4 en 6 variantes para short-term forecasting. El protocolo experimental estándar, establecido por Informer (Zhou et al., 2021), emplea ventana de entrada I = 96 timesteps y horizontes de predicción O ∈ {96, 192, 336, 720}, evaluando mediante Mean Squared Error (MSE) y Mean Absolute Error (MAE) sobre conjunto de test. TSLib distingue entre Look-Back-96 que fija ventana de entrada para comparabilidad directa, y Look-Back-Searching que permite optimización adaptativa.

Wang et al. (2025) establecen un hallazgo crítico que redefine expectativas de desempeño mediante la propuesta de una "Accuracy Law" que caracteriza objetivos de predicción e identifica datasets saturados. Los autores formalizan relación exponencial entre complejidad de patrones temporales y error mínimo alcanzable: MSE ≈ exp(α·Complexity), donde α es constante específica del dataset y Complexity cuantifica dificultad intrínseca mediante entropía de distribución temporal y estructura de periodicidad. La evidencia empírica demuestra saturación en múltiples benchmarks (ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity), donde métodos determinísticos actuales se aproximan asintóticamente a un límite teórico y mejoras incrementales mediante ajustes convencionales ofrecen retornos decrecientes.

Este hallazgo tiene implicaciones directas para el presente trabajo. Si tokenización HMM supera el límite MSE ≈ exp(α·Complexity) en benchmarks saturados, se validará empíricamente que estructura probabilística explícita representa cambio cualitativo necesario. La métrica de complejidad puede correlacionarse con número óptimo de estados K del HMM: series con mayor complejidad requerirían más estados para capturar regímenes heterogéneos, proporcionando fundamento teórico para selección de hiperparámetros. La saturación identificada no representa limitación técnica superable mediante ingeniería incremental, sino límite teórico intrínseco de enfoques determinísticos, posicionando la integración de HMM como dirección necesaria para avanzar más allá de este límite en el ecosistema estandarizado de TSLib. (https://github.com/thuml/Time-Series-Library)

### ✓ Marco teórico.

#### Hidden Markov Models: fundamento matemático

Como se estableció en el estado del arte, los HMM constituyen un marco probabilístico riguroso para modelización de secuencias con estados latentes. Rabiner (1989) sistematiza formalmente este framework, que se ha consolidado como referencia canónica para implementación y análisis teórico.

Un Hidden Markov Model se define mediante el conjunto de parámetros λ = (A, B, π):

- **A**: matriz de transición K×K entre estados ocultos con A_ij = P(q_t = j | q_{t-1} = i)
- **B**: distribuciones de emisión, típicamente gaussianas con parámetros (μ_k, σ_k) para cada estado k
- **π**: distribución inicial sobre estados con π_k = P(q_1 = k)

La probabilidad de una secuencia de observaciones O = (o₁, ..., o_T) se calcula marginalizando sobre todas las posibles secuencias de estados ocultos: P(O|λ) = Σ_Q P(O|Q, λ) P(Q|λ). Esta formulación permite capturar dependencias temporales explícitas mediante estructura markoviana donde el estado actual depende únicamente del estado previo según P(q_t | q_{t-1}), simplificando inferencia mediante programación dinámica. Hamilton (1989) demostró que esta propiedad es crítica para modelar cambios de régimen en series económicas.

#### Algoritmos fundamentales

El framework HMM se sustenta en tres algoritmos clásicos formalizados por Rabiner (1989) que resuelven los problemas canónicos de inferencia, decodificación y entrenamiento.

**Algoritmo de Viterbi (decodificación):** Encuentra la secuencia de estados óptima Q* = argmax_Q P(Q|O, λ) mediante programación dinámica:

- **Inicialización:** δ_1(i) = π_i · b_i(o_1)
- **Recursión:** δ_t(j) = max_i [δ_{t-1}(i) · a_{ij}] · b_j(o_t)
- **Backtracking:** recupera Q* mediante punteros almacenados

La complejidad es O(T·K²), lineal en longitud de secuencia y cuadrática en número de estados. Este algoritmo será crítico para asignar cada timestep a su estado latente correspondiente, generando la secuencia de tokens.

**Algoritmo de Baum-Welch (entrenamiento):** Estima parámetros λ* = argmax_λ P(O|λ) mediante Expectation-Maximization. Como establecen Dempster et al. (1977), EM garantiza convergencia monótona a máximo local alternando entre:

- **Paso E:** Calcula expectativas de ocupación γ_t(i) = P(q_t = i | O, λ) y transiciones ξ_t(i,j) = P(q_t = i, q_{t+1} = j | O, λ) usando forward-backward
- **Paso M:** Actualiza parámetros mediante:
  - π_i = γ_1(i)
  - a_{ij} = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
  - μ_i = Σ_t γ_t(i)·o_t / Σ_t γ_t(i)
  - σ_i² = Σ_t γ_t(i)·(o_t - μ_i)² / Σ_t γ_t(i)

Rabiner (1989) documenta técnicas de escalado numérico esenciales para evitar underflow con secuencias de miles de timesteps.

**Algoritmo Forward-Backward (inferencia):** Calcula probabilidades marginales P(q_t = k | O, λ) mediante dos pasadas:

- **Forward:** α_t(i) = P(o₁...o_t, q_t = i | λ) recursivo hacia adelante
- **Backward:** β_t(i) = P(o_{t+1}...o_T | q_t = i, λ) recursivo hacia atrás
- **Marginalización:** γ_t(i) = α_t(i)·β_t(i) / P(O|λ)

Estas probabilidades cuantifican incertidumbre en asignación de estados, proporcionando información complementaria a decodificación determinística de Viterbi con complejidad O(T·K²) idéntica.

#### HMM como mecanismo de embedding

La propuesta central consiste en reinterpretar estados ocultos como embeddings vectoriales estructurados. En aplicaciones tradicionales de reconocimiento de voz o bioinformática, HMM genera estados discretos z_t ∈ {1, 2, ..., K} utilizados directamente para clasificación. La innovación radica en utilizar parámetros del estado como embedding vectorial:

**Estado k → Embedding e_k = [μ_k, σ_k, A[k,:]]**

Esta representación encapsula simultáneamente tres tipos de información complementaria:

- **μ_k (media de emisión):** caracteriza valor central del régimen estadístico
- **σ_k (desviación estándar):** cuantifica volatilidad intrínseca del régimen (regímenes estables tienen σ_k baja, regímenes volátiles tienen σ_k alta)
- **A[k,:] (fila k de matriz de transición):** codifica dinámicas temporales mediante vector de K probabilidades donde A[k,k] alta indica régimen persistente y A[k,j] alta con j≠k indica transición frecuente hacia estado j

**Ejemplo ilustrativo:** HMM con K=3 estados aplicado a temperatura corporal medida horariamente. Tras entrenamiento con Baum-Welch, embeddings resultantes capturan regímenes fisiológicos diferenciados:

- **Estado 1 "Normal":** e₁ = [36.6, 0.3, [0.9, 0.08, 0.02]] representa normotermia con homeostasis estable (90% persistencia, 8% transición a febrícula, 2% fiebre súbita)
- **Estado 2 "Fiebre leve":** e₂ = [38.5, 0.5, [0.05, 0.25, 0.70]] representa régimen transitorio con tendencia a agravamiento (5% recuperación, 25% persistir, 70% progresión a fiebre alta)
- **Estado 3 "Fiebre alta":** e₃ = [39.8, 0.7, [0.15, 0.60, 0.25]] representa régimen crítico con múltiples trayectorias de recuperación (15% recuperación directa, 60% descenso gradual, 25% persistir)

Este ejemplo ilustra cómo embeddings HMM codifican simultáneamente información estadística local (μ_k, σ_k caracterizan régimen) y estructura temporal explícita (A[k,:] codifica dinámicas de transición). A diferencia de métodos determinísticos donde dependencia temporal es implícita (aprendida mediante attention) o inexistente (discretización sin memoria), HMM proporciona representación interpretable con fundamento probabilístico riguroso. La matriz A actúa como "memoria estructurada" que codifica explícitamente P(z_t | z_{t-1}) mediante estimación directa vía Baum-Welch, sin requerir aprendizaje de dependencias masivas como foundation models. Yeh & Tang (2022) demostraron que esta estructura explícita supera a métodos sin dependencias temporales en segmentación de audio discretizado.

#### Comparativa teórica con técnicas determinísticas

Las técnicas actuales presentan limitaciones en estructura probabilística, interpretabilidad o captura de dependencias temporales. La siguiente comparativa sistematiza diferencias fundamentales entre enfoques en términos de embedding, mecanismos de atención, tipo de dependencia temporal y ejemplos concretos

**Tabla 1. Comparativa de técnicas de tokenización para series temporales.**

| TÉCNICA | EMBEDDING | ATENCIÓN | DEPENDENCIA | EJEMPLO |
|---------|-----------|----------|-------------|---------|
| **Discretización** | 1. One-hot<br>2. Embedding matrix E∈ℝ^(K×D) | Sin attention (o self-attention si se añade Transformer) | **SIN MEMORIA**<br>Cada token independiente<br>P(tok_t) sin contexto | [20.4→23] y [23→20.4]<br>Ambas: [A,A]<br>Dirección perdida |
| **Text-based** | 1. BPE tokens LLM<br>2. Embeddings pre-entrenados<br>3. Positional encoding | Self-attention Transformer<br>Multi-head sobre tokens | **IMPLÍCITA**<br>Next-token prediction<br>Aprendida de corpus texto | "24.8" → ["24", ".", "8"]<br>LLM predice siguiente<br>usando patrones de texto |
| **Patching** | 1. Mean/Max pooling<br>2. Flatten concatenación<br>3. Proyección lineal W<br>4. CNN 1D | Self-attention inter-patch<br>Q(patch_i)·K(patch_j) | **DUAL**<br>Intra-patch: explícita (agregación)<br>Inter-patch: implícita (attention) | Patch [23.4,23.7,24.1,24.8]<br>Intra: mean=24.0<br>Inter: attention con siguiente patch |
| **Descomposición** | Por componente:<br>1. Trend: patches largos<br>2. Seasonal: FFT<br>3. Residual: L2 norm | Atención separada<br>por componente<br>o concatenados | **EXPLÍCITA**<br>por escala temporal<br>Trend: largo plazo<br>Seasonal: cíclico | Trend: subida +1°C/día<br>Seasonal: pico verano<br>Cada uno su dependencia |
| **Foundation** | 1. RevIN normalización<br>2. Patches fixed/adaptive<br>3. Masked 30%<br>4. Embeddings ℝ^768 | Multi-head self-attention<br>Transformer estándar<br>Aprendido en 27B+ timesteps | **IMPLÍCITA masiva**<br>Emerge de escala<br>Masked reconstruction<br>fuerza contexto | Input: [P₁,[MASK],P₃]<br>Target: reconstruir P₂<br>Aprende dependencias<br>de millones de series |
| **HMM** | Estado k → e_k = [μ_k, σ_k, A[k,:]] | Transiciones probabilísticas (Viterbi) para tokenización + multi-head self-attention para predicción | **EXPLÍCITA**<br>Matriz A transiciones<br>P(z_t\|z_{t-1}) dinámica | Estado 1: [36.6, 0.3, [0.9,0.08,0.02]]<br>Estado 2: [38.5, 0.5, [0.05,0.25,0.70]]<br>A[1,1]=0.9: 90% persistir |

**Tabla 2. Características de granularidad, adaptabilidad y memoria.**

| Método | Granularidad | Adaptabilidad | Memoria |
|--------|--------------|---------------|---------|
| **Discretización** | Fina (1:1) | Fija | SIN memoria |
| **Text-based** | Muy fina | Semi | Implícita |
| **Patching (P=4)** | Media (P:1) | Fija | Implícita |
| **Descomposición** | Gruesa | Semi | Por componente |
| **Foundation** | Media | Aprendida | Implícita masiva |
| **HMM** | Adaptativa | Dinámica | **EXPLÍCITA** |

Las tablas evidencian que HMM constituye el único enfoque con dependencia temporal explícita mediante estructura probabilística P(z_t|z_{t-1}) estimada directamente, granularidad adaptativa que se ajusta dinámicamente a cambios de régimen sin ventanas fijas, y embeddings interpretables donde cada dimensión tiene significado estadístico concreto. Esta combinación fundamenta teóricamente el potencial del HMM para superar limitaciones de métodos determinísticos identificadas en el estado del arte.

---

## OBJETIVO GENERAL Y OBJETIVOS ESPECÍFICOS

*El objetivo general debe ser la acción a realizar para responder el problema de investigación planteado en el punto anterior. Los objetivos específicos deben ser las acciones a realizar para responder las preguntas de investigación planteadas.*

### Objetivo general

Desarrollar e implementar un sistema de tokenización de series temporales basado en Estados Ocultos de Markov, donde los estados ocultos actúen como embeddings latentes estructurados, evaluar su desempeño en tareas de predicción a largo plazo frente a técnicas determinísticas actuales, y analizar los trade-offs entre complejidad computacional, ratio de compresión e interpretabilidad de regímenes.

### Objetivos específicos

**OE1 (RQ1):** Realizar una revisión sistemática de las técnicas actuales de tokenización y embedding para series temporales en el contexto de LLMs, identificando sus limitaciones en la captura de dependencias temporales y regímenes estadísticos.

**OE2 (RQ2):** Formalizar el marco teórico de HMM como mecanismo de representación latente, caracterizando las propiedades probabilísticas que permiten modelar cambios de régimen y dependencias temporales explícitas.

**OE3 (RQ3):** Diseñar e implementar un pipeline completo de tokenización basado en HMM que transforme series temporales continuas en embeddings vectoriales e_k = [μ_k, σ_k, A[k,:]], integrando RevIN para normalización, Baum-Welch para entrenamiento y Viterbi para decodificación.

**OE4 (RQ4):** Establecer un protocolo de evaluación experimental que mida preservación de información, capacidad de reconstrucción y desempeño en forecasting univariado mediante métricas MSE y MAE sobre datasets benchmark (ETTh1, ETTh2, Weather, Electricity) y zero-shot (Traffic, Exchange).

**OE5 (RQ5):** Cuantificar los trade-offs entre complejidad computacional (tiempo de entrenamiento/inferencia), ratio de compresión (número de tokens generados) y desempeño predictivo, comparando HMM con baselines determinísticos en horizontes de predicción {96, 192, 336, 720}.

---

## METODOLOGÍA:

### Pipeline propuesto

El sistema RITMO implementa un pipeline de cinco etapas que integra normalización reversible, modelado probabilístico HMM y predicción mediante transformer. Este diseño modular permite validar empíricamente la efectividad de los embeddings estructurados e_k = [μ_k, σ_k, A[k,:]] propuestos en el marco teórico.

#### Etapa 1: Normalización RevIN

Se aplica Reversible Instance Normalization (Kim et al., 2022) a cada serie temporal X de forma independiente:

**X_norm = (X - μ) / σ**

donde μ y σ son la media y desviación estándar calculadas sobre la serie completa. Los parámetros (μ, σ) se almacenan para desnormalización posterior. Esta técnica garantiza estacionariedad (media = 0, varianza = 1) y robustez frente a distribution shift, requisitos críticos para correcto funcionamiento de HMM con emisiones gaussianas. Liu et al. (2022) demuestran que normalización facilita convergencia del algoritmo EM y mejora estabilidad numérica en estimación de parámetros.

#### Etapa 2: Entrenamiento HMM (Baum-Welch)

Sobre las series normalizadas del conjunto de entrenamiento {ETTh1, ETTh2, Weather, Electricity}, se estima un único modelo HMM con K estados mediante algoritmo Baum-Welch que maximiza verosimilitud P(O|λ) iterativamente:

1. **Inicialización:** Parámetros λ⁽⁰⁾ = (A⁽⁰⁾, B⁽⁰⁾, π⁽⁰⁾) mediante k-means sobre observaciones, siguiendo estrategia estándar de Rabiner (1989)
2. **Paso E:** Forward-backward calcula probabilidades marginales γ_t(k) = P(q_t=k|O,λ) y de transición ξ_t(k,l) = P(q_t=k, q_{t+1}=l|O,λ)
3. **Paso M:** Actualiza parámetros mediante ecuaciones de re-estimación: π_k = γ_1(k), A_kl = Σ_t ξ_t(k,l) / Σ_t γ_t(k), μ_k = Σ_t γ_t(k)·o_t / Σ_t γ_t(k), σ²_k = Σ_t γ_t(k)·(o_t - μ_k)² / Σ_t γ_t(k)
4. **Convergencia:** Se itera hasta que |log P(O|λ⁽ⁿ⁾) - log P(O|λ⁽ⁿ⁻¹⁾)| < ε

El modelo resultante λ* = (A*, B*, π*) se guarda junto con parámetros RevIN (μ, σ) de cada dataset de entrenamiento. Este modelo frozen se aplicará posteriormente a datasets zero-shot sin re-entrenamiento, validando capacidad de generalización de los embeddings aprendidos.

#### Etapa 3: Tokenización (Viterbi)

Para cada serie de test (incluyendo zero-shot Traffic y Exchange), se aplica RevIN con parámetros guardados correspondientes y se ejecuta algoritmo de Viterbi sobre HMM entrenado para obtener secuencia óptima de estados ocultos:

**Q* = argmax_Q P(Q|O, λ*)**

Como se formalizó en el marco teórico, Viterbi opera mediante programación dinámica con complejidad O(T·K²), calculando recursivamente variables δ_t(k) que representan máxima probabilidad de alcanzar estado k en timestep t. La secuencia resultante Q* = [z₁, z₂, ..., z_T] constituye la tokenización de la serie, donde cada z_t ∈ {1, 2, ..., K} representa el estado activo en el timestep t.

#### Etapa 4: Generación de embeddings

Cada estado k se transforma en embedding vectorial e_k ∈ ℝ^(2+K) mediante la representación propuesta:

**e_k = [μ_k, σ_k, A[k,:]]**

donde μ_k ∈ ℝ es media de emisión (centro del régimen), σ_k ∈ ℝ es desviación estándar (dispersión del régimen), y A[k,:] ∈ ℝ^K es fila k de matriz transición (probabilidades de transición a cada estado). La secuencia de tokens [z₁, ..., z_T] se mapea a embeddings [e_{z₁}, ..., e_{z_T}], generando representación estructurada que encapsula simultáneamente información estadística local y dinámica temporal explícita. Esta transformación constituye la innovación central del pipeline RITMO.

#### Etapa 5: Predicción con Transformer

Los embeddings se procesan mediante transformer decoder que realiza autoregressive forecasting:

**ŷ_norm = Transformer([e_{z₁}, ..., e_{z_I}])**

donde I = 96 es longitud de input y ŷ_norm ∈ ℝ^O es predicción normalizada para horizontes O ∈ {96, 192, 336, 720}. La arquitectura específica del transformer (número de capas, attention heads, dimensión oculta) se determinará mediante validación en fase de implementación, considerando configuraciones baseline de la literatura (e.g., 4 capas, 8 heads, dim=512). Finalmente, se aplica RevIN inverso para recuperar escala original:

**ŷ = ŷ_norm × σ + μ**

Esta arquitectura permite que transformer opere sobre representaciones de alto nivel (regímenes estadísticos) en lugar de valores crudos, potencialmente facilitando captura de patrones de largo plazo mediante estructura temporal explícita codificada en A[k,:].

### Datasets

Como se estableció en el estado del arte, Time-Series-Library (TSLib) constituye el repositorio de referencia que consolida los benchmarks estándar de la comunidad científica. Los datasets están disponibles públicamente en formato pre-procesado, garantizando reproducibilidad experimental. Se emplean seis datasets divididos en dos grupos según su función en el pipeline:

#### Grupo 1 - Entrenamiento HMM (4 datasets):

- **ETTh1 (Zhou et al., 2021):** Transformer de temperatura oil (OT) con frecuencia horaria. Split 7:1:2 (train/val/test). Benchmark estándar universal empleado consistentemente en tablas comparativas de Informer, PatchTST y TimesNet.
- **ETTh2 (Zhou et al., 2021):** Variante de ETTh1 con distribution shift documentado entre splits. Valida robustez de RevIN ante cambios distribucionales, crítico para evaluar generalización del HMM a regímenes no vistos durante entrenamiento.
- **Weather (Zhou et al., 2021):** Temperatura wet-bulb (WetBulbCelsius) con regímenes climáticos estacionales claros (invierno/verano). Ideal para evaluar si estados ocultos del HMM capturan cambios de régimen interpretables correspondientes a estaciones del año.
- **Electricity (Trindade, 2015; Lai et al., 2018):** Consumo eléctrico (MT_320) con periodicidad diaria fuerte (24h). Procesado desde UCI ML Repository. Valida detección de patrones cíclicos mediante estados que representen regímenes día/noche.

#### Grupo 2 - Evaluación Zero-shot (2 datasets):

- **Traffic (Lai et al., 2018):** Tasa de ocupación de sensores de tráfico (1 sensor univariado). Dominio diferente de datasets de entrenamiento, con regímenes rush-hour explícitos. Se evalúa HMM frozen (sin re-entrenamiento) para medir generalización a series con estructura temporal similar pero estadísticas diferentes.
- **Exchange (Lai et al., 2018):** Tipo de cambio (1 de 8 países). Series financieras sin periodicidad marcada, constituyendo test de robustez para HMM entrenado en series con estacionalidad clara. Evalúa si embeddings generalizan a dominios con regímenes menos estructurados.

Todos los datasets emplean split 7:1:2 (train/val/test) excepto Traffic y Exchange, que utilizan únicamente split test para evaluación zero-shot, siguiendo protocolo de generalización documentado en TSLib.

### Configuración experimental

El protocolo experimental sigue el estándar establecido por Informer (Zhou et al., 2021) y adoptado por TSLib:

- **Input length:** I = 96 timesteps (Look-Back-96 en nomenclatura TSLib)
- **Prediction horizons:** O ∈ {96, 192, 336, 720} timesteps
- **Métricas primarias:** MSE (Mean Squared Error), MAE (Mean Absolute Error)
- **Reporte:** Promedio sobre los cuatro horizontes de predicción
- **Número de estados HMM:** A determinar mediante selección por validación cruzada con criterios AIC/BIC (Akaike Information Criterion, Bayesian Information Criterion)
- **Transformer:** Probablemente, arquitectura decoder-only con configuración a determinar en fase de implementación (configuración inicial de referencia: 4 capas, 8 attention heads, dimensión oculta 512)

### Comparación con baselines

Los resultados del pipeline RITMO se compararán con cuatro métodos representativos del estado del arte, permitiendo cuantificar ganancias atribuibles específicamente a tokenización HMM:

- **PatchTST (Nie et al., 2023):** Baseline principal que implementa patching con channel-independence, segmentando series en patches de longitud fija (típicamente P=16) que actúan como tokens de entrada. Reduce complejidad de atención de O(L²) a O((L/S)²) y permite ventanas de lookback largas. Su desempeño estado del arte en múltiples benchmarks lo establece como referencia natural para comparación.

- **DLinear (Zeng et al., 2023):** Baseline simple que descompone mediante promedio móvil y aplica capas lineales separadas a trend y seasonal. Zeng et al. demuestran sorprendentemente que este enfoque simple supera a transformers complejos en múltiples benchmarks, cuestionando necesidad de arquitecturas sofisticadas y sugiriendo que estructura temporal bien capturada puede ser más crítica que complejidad arquitectural. Establece referencia de desempeño mínimo para validar que mejoras arquitectónicas justifican su complejidad.

- **TimeMixer (Wang et al., 2024b):** Baseline reciente que implementa descomposición multiscale mixing aplicando mixing operations separadamente a diferentes escalas temporales (past-decomposable-mixing para patrones históricos, future-multipredictor-mixing para horizontes distintos). Esta arquitectura representa estado del arte en modelado de dependencias multi-resolución, permitiendo evaluar si estructura markoviana explícita de HMM puede competir con descomposición multi-escala sofisticada.

- **TimeXer (Wang et al., 2024a):** Baseline que incorpora explícitamente variables exógenas conocidas mediante exogenous-aware attention que modula atención endógena con información externa. Aunque el presente trabajo se enfoca en forecasting univariado sin covariates, TimeXer proporciona referencia de desempeño alcanzable cuando información externa estructurada se integra explícitamente, permitiendo posicionar HMM (que estructura información temporal internamente) frente a métodos que estructuran información externa.

Todos los baselines se entrenarán con hiperparámetros óptimos reportados en publicaciones originales, garantizando comparación justa.

### Limitaciones del scope experimental

Aunque el estado del arte identifica cinco categorías de tokenización, la evaluación se restringe a métodos con implementaciones reproducibles y benchmarks univariados consolidados. Se excluyen tres categorías por razones metodológicas:

1. **Discretización moderna:** TOTEM (Talukder et al., 2024) y VQ-VAE (van den Oord et al., 2017) están diseñados para forecasting multivariado, donde cuantización opera sobre espacio conjunto de variables. En configuración univariada degeneran a binning escalar sin propiedades de compresión vectorial. No existen implementaciones ni benchmarks univariados publicados sobre ETT/Weather/Electricity.

2. **Text-based:** Chronos (Ansari et al., 2024) y LLMTime (Gruver et al., 2023), propuestas recientes que verbalizan series como strings para LLMs, no han sido adoptados como baselines por la comunidad—ausentes en tablas comparativas posteriores (TimesNet, PatchTST, FPT). Chronos emplea vocabulario discreto de 4096 símbolos, cuya comparación con embeddings continuos HMM requeriría quantization-aware training fuera del alcance.

3. **Foundation models:** MOMENT (Goswami et al., 2024) pre-entrenado sobre 27B+ timesteps introduce confounders: (a) con pesos pre-entrenados, diferencias reflejan datos masivos (no controlables) además de arquitectura; (b) re-entrenar desde cero tiene costo prohibitivo (≈1000 GPU-horas) y elimina ventaja fundamental del paradigma. FPT (Zhou et al., 2023) introduce el mismo confounder.

La comparación se centra en cuatro baselines con: (a) implementaciones reproducibles sin pre-entrenamiento masivo, (b) presencia consistente en benchmarks univariados TSLib, y (c) representación de paradigmas arquitectónicos principales.

### Protocolo de evaluación

El análisis comparativo seguirá cuatro dimensiones complementarias:

1. **Desempeño predictivo:** MSE y MAE promediados sobre horizontes O ∈ {96, 192, 336, 720} con I=96 fijo, reportando resultados por dataset y promedio general (formato Tabla 2 de Wu et al., 2023). Valida si tokenización HMM mejora error predictivo comparado con métodos determinísticos, especialmente en horizontes largos donde estructura temporal explícita A[k,:] puede proporcionar ventajas.

2. **Ratio de compresión:** Timesteps originales dividido entre tokens generados. Para HMM corresponde al número de cambios de estado detectados por Viterbi; para PatchTST es T/P con P=16. Cuantifica eficiencia representacional: ratios altos indican captura de patrones mediante pocos tokens, reduciendo complejidad downstream.

3. **Eficiencia computacional:** Tiempo de entrenamiento e inferencia en GPU NVIDIA RTX 4090 (24GB VRAM), reportando tiempo promedio por epoch, por muestra y throughput (muestras/segundo). Cuantifica trade-off entre desempeño y costo, validando si mejoras MSE/MAE justifican overhead de Baum-Welch y Viterbi.

4. **Interpretabilidad de regímenes:** Análisis cualitativo exclusivo para HMM inspeccionando parámetros (μ_k, σ_k, A) mediante: (a) distribución temporal de activaciones (¿estados con periodicidad 24h en Electricity?), (b) interpretación de μ_k/σ_k (¿regímenes temperatura/consumo/volatilidad alta/baja?), y (c) estructura de transiciones en A (¿estados persistentes A[k,k]>>0 vs. transitorios?). Ejemplos esperados: día/noche en Electricity, estacionalidad en Weather, bull/bear en Exchange. Valida hipótesis de interpretabilidad superior frente a representaciones latentes sin significado estadístico explícito.

### Limitaciones y trabajo futuro

El enfoque propuesto presenta limitaciones inherentes que abren direcciones de investigación complementarias. En primer lugar, la asunción de emisiones gaussianas B_k(o_t) = N(o_t; μ_k, σ²_k) puede resultar restrictiva para series temporales con distribuciones no-normales como colas pesadas o multimodalidad dentro de un mismo régimen. Fox et al. (2011) abordan este problema mediante emisiones no paramétricas basadas en Dirichlet Process mixtures of Gaussians en el contexto de sticky HDP-HMM, permitiendo que cada estado capture distribuciones arbitrariamente complejas. La integración de esta extensión en el pipeline RITMO constituye dirección natural para incrementar capacidad representacional.

En segundo lugar, el número de estados K se fija como hiperparámetro determinado mediante validación cruzada, requiriendo búsqueda exhaustiva computacionalmente costosa. La extensión a modelos no-paramétricos Bayesianos como HDP-HMM (Hierarchical Dirichlet Process HMM) permitiría inferir automáticamente número óptimo de estados directamente de los datos mediante inferencia variacional, como demuestran Fox et al. (2011) en aplicaciones de diarización de hablantes.

En tercer lugar, la asunción markoviana de primer orden P(z_t | z_{t-1}) impone memoria limitada de un único timestep previo, potencialmente insuficiente para capturar dependencias de largo alcance. Los Hidden Semi-Markov Models (HSMM) extienden HMM permitiendo que cada estado persista durante duraciones variables modeladas explícitamente mediante distribuciones de duración. Dai et al. (2017) proponen R-HSMM que combina HSMM con RNNs generativas para emisiones no-lineales, demostrando superioridad en segmentación de secuencias complejas.

En cuarto lugar, el presente trabajo se limita a forecasting univariado donde cada serie se procesa independientemente. La extensión a forecasting multivariado requeriría modelar dependencias cross-channel además de dependencias temporales. Una dirección prometedora consiste en combinar HMM para tokenización temporal con mecanismos de atención cross-dimensional como los propuestos en Crossformer (Zhang & Aggarwal, 2023), permitiendo captura simultánea de transiciones de régimen dentro de cada variable y correlaciones dinámicas entre variables.

Finalmente, el entrenamiento mediante Baum-Welch converge a máximos locales dependiendo de inicialización. Mensch & Blondel (2018) proponen hacer diferenciables algoritmos de programación dinámica mediante suavizado del operador max, permitiendo integración end-to-end con gradientes backpropagados desde pérdida de predicción downstream. Esta extensión permitiría entrenar conjuntamente HMM y Transformer maximizando directamente desempeño predictivo en lugar de verosimilitud de secuencias observadas, potencialmente mejorando generalización.

Este trabajo emplea RevIN (Kim et al., 2022) para normalización, siguiendo protocolo estándar de TSLib y garantizando comparabilidad directa con baselines. Trabajos recientes como Inner-instance normalization (Jibao et al., 2025) proponen normalización point-level que detecta distribution shifts dentro de instancias individuales. Explorar si esta técnica mejora detección de cambios de régimen en HMM constituye dirección futura prometedora, ya que podría capturar transiciones de estado más finas al eliminar shifts locales que actualmente se confunden con cambios de régimen estadístico.

---

## CRONOGRAMA:

### Fase 1: Marco teórico y diseño metodológico

**Duración:** septiembre 2025 - diciembre 2025 (4 meses)

**Entregables:**
- Anteproyecto aprobado
- Revisión bibliográfica completa
- Diseño del pipeline RITMO

### Fase 2: Ingeniería del Dato

**Duración:** diciembre 2025 - febrero 2026 (3 meses)

**Entregables:**
- Datasets benchmark preparados y normalizados (ETTh1, ETTh2, Weather, Electricity, Traffic, Exchange)
- Pipeline de preprocesamiento (RevIN, splits train/val/test)
- Implementación completa de algoritmos HMM (Baum-Welch, Viterbi, Forward-Backward)
- HMM entrenado sobre 4 datasets

### Fase 3: Análisis de los Datos

**Duración:** enero 2026 - marzo 2026 (3 meses)

**Entregables:**
- Generación de embeddings estructurados e_k = [μ_k, σ_k, A[k,:]]
- Pipeline completo de forecasting con Transformer
- Experimentos sobre 6 datasets (4 benchmark + 2 zero-shot)
- Resultados comparativos con 6 baselines (MSE, MAE en horizontes {96, 192, 336, 720})

### Fase 4: Análisis de Negocio

**Duración:** febrero 2026 - abril 2026 (3 meses)

**Entregables:**
- Análisis de trade-offs: complejidad computacional vs. desempeño
- Cuantificación de ratio de compresión
- Interpretabilidad de regímenes estadísticos (análisis cualitativo de estados ocultos)

### Fase 5: Redacción y defensa

**Duración:** mayo 2026 - junio 2026 (2 meses)

**Entregables:**
- Memoria final del TFG
- Presentación y defensa ante tribunal

**Duración total:** 10 meses (septiembre 2025 - junio 2026)

**Nota:** La redacción de la memoria se desarrolla de forma continua a lo largo de todas las fases, consolidándose en la Fase 5.

**Nota:** Las referencias bibliográficas, datasets, métricas, configuraciones experimentales y metodologías específicas descritas en este anteproyecto son preliminares y especulativas, basadas en la revisión del estado del arte actual; estas se ajustarán y refinirán durante el desarrollo del proyecto según surjan nuevos problemas, limitaciones técnicas o avances relevantes en la literatura.

---

## REFERENCIAS

### Técnicas de Tokenización

#### Discretización

Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Pineda Arango, S., Kapoor, S., Zschiegner, J., Maddix, D. C., Wang, H., Mahoney, M. W., Torkkola, K., Wilson, A. G., Bohlke-Schneider, M., & Wang, Y. (2024). Chronos: Learning the language of time series. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=gerNCVqqtR

Lin, J., Keogh, E., Wei, L., & Lonardi, S. (2007). Experiencing SAX: A novel symbolic representation of time series. *Data Mining and Knowledge Discovery*, *15*(2), 107-144. https://doi.org/10.1007/s10618-007-0064-z

Talukder, S., Yue, Y., & Gkioxari, G. (2024). TOTEM: TOkenized time series embeddings for general time series analysis. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=QlTLkH6xRC

van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural discrete representation learning. *Advances in Neural Information Processing Systems*, *30*, 6306-6315. https://arxiv.org/abs/1711.00937

Zhao, Y., Zhou, T., Chen, C., Sun, L., Qian, Y., & Jin, R. (2024). *Sparse-VQ Transformer: An FFN-Free Framework with Vector Quantization for Enhanced Time Series Forecasting* (arXiv:2402.05830) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2402.05830

#### Patching

Abeywickrama, S., Eldele, E., Wu, M., Li, X., & Yuen, C. (2025). EntroPE: Entropy-guided dynamic patch encoder for time series forecasting. *arXiv preprint arXiv:2509.26157*. https://arxiv.org/abs/2509.26157

Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. *Proceedings of the 11th International Conference on Learning Representations*. https://arxiv.org/abs/2211.14730

Peršak, E., Anjos, M. F., Lautz, S., & Kolev, A. (2024). Multiple-resolution tokenization for time series forecasting with an application to pricing. *arXiv preprint arXiv:2407.03185*. https://arxiv.org/abs/2407.03185

Zhang, L., & Aggarwal, C. (2023). Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. *Proceedings of the 11th International Conference on Learning Representations*. https://openreview.net/forum?id=vSVLM2j9eie

Zhang, Y., Ma, L., Pal, S., Zhang, Y., & Coates, M. (2024). Multi-resolution time-series transformer for long-term forecasting. *Proceedings of the 27th International Conference on Artificial Intelligence and Statistics* (Vol. 238). PMLR. https://arxiv.org/abs/2311.04147

#### Descomposición

Cao, D., Jia, F., Arik, S. Ö., Pfister, T., Zheng, Y., Ye, W., & Liu, Y. (2024). TEMPO: Prompt-based generative pre-trained transformer for time series forecasting. *The Twelfth International Conference on Learning Representations*. https://openreview.net/forum?id=YH5w12OUuU

Woo, G., Liu, C., Sahoo, D., Kumar, A., & Hoi, S. (2022). CoST: Contrastive learning of disentangled seasonal-trend representations for time series forecasting. *International Conference on Learning Representations*. https://openreview.net/forum?id=PilZY3omXV2

Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. *Advances in Neural Information Processing Systems* (Vol. 34, pp. 22419-22430). https://proceedings.neurips.cc/paper_files/paper/2021/file/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html

Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., & Jin, R. (2022). FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. *Proceedings of the 39th International Conference on Machine Learning* (Vol. 162, pp. 27268-27286). PMLR. https://proceedings.mlr.press/v162/zhou22g.html

#### Foundation Models

Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., & Dubrawski, A. (2024). MOMENT: A family of open time-series foundation models. *Proceedings of the 41st International Conference on Machine Learning* (Vol. 235, pp. 16115-16152). PMLR. https://proceedings.mlr.press/v235/goswami24a.html

Liu, Y., Zhang, H., Li, C., Huang, X., Wang, J., & Long, M. (2024). Timer: Generative pre-trained transformers are large time series models. *Proceedings of the 41st International Conference on Machine Learning*. https://arxiv.org/abs/2402.02368

Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., & Sahoo, D. (2024). Unified training of universal time series forecasting transformers. *Proceedings of the 41st International Conference on Machine Learning* (Vol. 235, pp. 53140-53164). PMLR. https://proceedings.mlr.press/v235/woo24a.html

Zhou, T., Niu, P., Wang, X., Sun, L., & Jin, R. (2023). One fits all: Power general time series analysis by pretrained LM. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. https://arxiv.org/abs/2302.11939

#### Text-based

Gruver, N., Finzi, M., Qiu, S., & Wilson, A. G. (2023). Large language models are zero-shot time series forecasters. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. https://arxiv.org/abs/2310.07820

Jin, M., Wang, S., Ma, L., Chu, Z., Zhang, J. Y., Shi, X., Chen, P.-Y., Liang, Y., Li, Y.-F., Pan, S., & Wen, Q. (2024). Time-LLM: Time series forecasting by reprogramming large language models. *arXiv preprint arXiv:2402.10835*. https://arxiv.org/abs/2402.10835

#### Transformers Eficientes y Baselines Consolidados

Wang, Y., Wu, H., Dong, J., Liu, Y., Qiu, Y., Zhang, H., Wang, J., & Long, M. (2024). TimeXer: Empowering transformers for time series forecasting with exogenous variables. In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. https://proceedings.neurips.cc/paper_files/paper/2024/file/0113ef4642264adc2e6924a3cbbdf532-Paper-Conference.pdf

Wang, Y., Wu, H., Dong, J., Liu, Y., Long, M., & Wang, J. (2024). TimeMixer: Decomposable multiscale mixing for time series forecasting. *Proceedings of the 12th International Conference on Learning Representations (ICLR 2024)*. https://openreview.net/forum?id=7oLshfEIC2

Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., & Long, M. (2023). TimesNet: Temporal 2D-variation modeling for general time series analysis. *International Conference on Learning Representations*. https://openreview.net/forum?id=ju_Uqw384Oq

Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are transformers effective for time series forecasting? *Proceedings of the AAAI Conference on Artificial Intelligence*, *37*(9), 11121-11128. https://doi.org/10.1609/aaai.v37i9.26317

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, *35*(12), 11106-11115. https://doi.org/10.1609/aaai.v35i12.17325

### Surveys

Abdullahi, S., Danyaro, K. U., Zakari, A., Aziz, I. A., Zawawi, N. A. W. A., & Adamu, S. (2025). Timeseries large language models: A systematic review of state-of-the-art. *IEEE Access*, *13*, 30235-30261. https://doi.org/10.1109/ACCESS.2025.3535782

Jiang, Y., Pan, Z., Zhang, X., Garg, S., Schneider, A., Nevmyvaka, Y., & Song, D. (2024). Empowering time series analysis with large language models: A survey. *arXiv preprint arXiv:2402.03182*. https://doi.org/10.48550/arXiv.2402.03182

Liang, Y., Wen, H., Nie, Y., Jiang, Y., Jin, M., Song, D., Pan, S., & Wen, Q. (2024). Foundation models for time series analysis: A tutorial and survey. *Proceedings of the 30th ACM SIGKKDD Conference on Knowledge Discovery and Data Mining* (pp. 6325-6336). ACM. https://doi.org/10.1145/3637528.3671451

Wen, Q., Zhou, T., Zhang, C., Chen, W., Ma, Z., Yan, J., & Sun, L. (2023). Transformers in time series: A survey. *Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI 2023)*. https://www.ijcai.org/proceedings/2023/0759.pdf

Zhang, X., Chowdhury, R. R., Gupta, R. K., & Shang, J. (2024). Large language models for time series: A survey. *arXiv preprint arXiv:2402.01801*. https://arxiv.org/abs/2402.01801

### Preprocesamiento y Normalización

Jibao, Z., Fu, Y., Chen, X., & Chen, G. (2025). Inner-instance normalization for time series forecasting. *arXiv preprint arXiv:2510.08657*. https://arxiv.org/abs/2510.08657

Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022). Reversible instance normalization for accurate time-series forecasting against distribution shift. *International Conference on Learning Representations*. https://openreview.net/forum?id=cGDAkQo1C0p

Liu, Y., Wu, H., Wang, J., & Long, M. (2022). Non-stationary transformers: Exploring the stationarity in time series forecasting. *Advances in Neural Information Processing Systems* (Vol. 35, pp. 9881-9893). https://proceedings.neurips.cc/paper_files/paper/2022/file/4054556fcaa934b0bf76da52cf4f92cb-Paper-Conference.pdf

### HIDDEN MARKOV MODELS

#### Fundamentos Teóricos

Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Methodological)*, *39*(1), 1-38. https://doi.org/10.1111/j.2517-6161.1977.tb01600.x

Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, *77*(2), 257-286. https://doi.org/10.1109/5.18626

#### Regime-Switching y Estructura Temporal

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, *57*(2), 357-384. https://doi.org/10.2307/1912559

Tang, B., & Matteson, D. S. (2021). Probabilistic transformer for time series analysis. In *Advances in Neural Information Processing Systems 34 (NeurIPS 2021)* (pp. 23592-23608). https://proceedings.neurips.cc/paper/2021/file/c68bd9055776bf38d8fc43c0ed283678-Paper.pdf

#### Selección Automática de Estados

Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2011). A sticky HDP-HMM with application to speaker diarization. *The Annals of Applied Statistics*, *5*(2A), 1020-1056. https://doi.org/10.1214/10-AOAS395

#### Aplicaciones Modernas

Dai, H., Dai, B., Zhang, Y.-M., Li, S., & Song, L. (2017). Recurrent hidden semi-Markov model. In *Proceedings of the International Conference on Learning Representations (ICLR 2017)*. https://openreview.net/forum?id=HkwVAXyCb

Mensch, A., & Blondel, M. (2018). Differentiable dynamic programming for structured prediction and attention. In *Proceedings of the 35th International Conference on Machine Learning* (Vol. 80, pp. 3462-3471). PMLR. https://proceedings.mlr.press/v80/mensch18a.html

Yeh, S.-L., & Tang, H. (2022). Learning dependencies of discrete speech representations with neural hidden Markov models. *arXiv preprint arXiv:2210.16659*. https://doi.org/10.48550/arXiv.2210.16659

### Datasets

Lai, G., Chang, W.-C., Yang, Y., & Liu, H. (2018). Modeling long- and short-term temporal patterns with deep neural networks. *Proceedings of the 41st International ACM SIGIR Conference on Research & Development in Information Retrieval* (pp. 95-104). ACM. https://doi.org/10.1145/3209978.3210006

Trindade, A. (2015). *ElectricityLoadDiagrams20112014 Data Set*. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

Wang, Y., Wu, H., Ma, Y., Fang, Y., Zhang, Z., Liu, Y., Wang, S., Ye, Z., Xiang, Y., Wang, J., & Long, M. (2025). Accuracy law for the future of deep time series forecasting. *arXiv preprint arXiv:2510.02729*. https://arxiv.org/abs/2510.02729
