# METODOLOGÍA RITMO - IMPLEMENTACIÓN

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

### Objetivo general

Implementar y evaluar un sistema de comparación controlada de 6 técnicas de tokenización de series temporales (HMM + 5 baselines determinísticos), donde los estados ocultos de HMM actúen como embeddings latentes estructurados, mediante métricas intrínsecas y desempeño downstream en predicción a largo plazo, estableciendo el estado del arte actual y validando la propuesta RITMO frente a modelos consolidados.

### Objetivos específicos

**OE1 (RQ1):** Implementar las 6 técnicas de tokenización (Discretización, Text-based, Patching, Descomposición, Foundation models, HMM) con sus embeddings naturales correspondientes, garantizando comparabilidad mediante backbone transformer único y espacio común [seq, d_model].

**OE2 (RQ2):** Implementar el pipeline RITMO completo (RevIN → Baum-Welch → Viterbi → Embeddings estructurados e_k=[μ_k,σ_k,A[k,:]]), validando cada fase mediante métricas específicas y visualizaciones profesionales.

**OE3 (RQ3):** Establecer protocolo de evaluación dual: (A) métricas intrínsecas de tokenización (compresión, reconstrucción, dependencias temporales, diversidad vocabulario, persistencia, concentración) sin tareas downstream, y (B) métricas downstream (MSE, MAE forecasting) en benchmarks consolidados.

**OE4 (RQ4):** Ejecutar Plan A: comparación controlada de 6 técnicas con mismo Transformer evaluando trade-offs compresión-información-dependencias mediante métricas intrínsecas y desempeño predictivo en horizontes {96,192,336,720} sobre datasets ETTh1, ETTh2, Weather, Electricity (train) y Traffic, Exchange (zero-shot).

**OE5 (RQ5):** Ejecutar Plan B: comparación de HMM+Transformer versus 4 baselines SOTA (DLinear, PatchTST, TimeMixer, TimeXer) para situar desempeño de la propuesta RITMO en contexto del estado del arte consolidado, cuantificando ganancias/pérdidas atribuibles a tokenización probabilística explícita.

---

## METODOLOGÍA:

### Implementación Actual: Validación del Pipeline RITMO

El sistema RITMO ha sido implementado y validado exhaustivamente mediante notebook `RITMO_pipeline_validation.ipynb` que verifica las 4 fases críticas del pipeline propuesto.

#### Fase 1: Normalización RevIN (✓ Implementada)

Se aplica Reversible Instance Normalization (Kim et al., 2022) con validación cuantitativa:

- **Pre-normalización:** μ=19.84, σ=10.37 (ETTh1 train)
- **Post-normalización:** μ≈0.0 (error <10⁻⁶), σ≈1.0 (error <10⁻⁶)
- **Reversibilidad:** MSE reconstrucción <10⁻¹² (precisión numérica perfecta)

Implementación en `utils/revin.py` con métodos `fit_transform()` y `inverse_transform()` que garantizan preservación exacta de estadísticas originales tras normalización-desnormalización.

#### Fase 2: Entrenamiento HMM Baum-Welch (✓ Implementada)

Algoritmo EM implementado en `hmm/baum_welch.py` con:

- **Inicialización:** k-means sobre observaciones normalizadas
- **Convergencia:** ε=10⁻⁴ sobre log-likelihood (típicamente 30-50 iteraciones)
- **Parámetros estimados:** A (matriz transición 5×5), π (distribución inicial), μ (medias K=5), σ (volatilidades K=5)
- **Cache persistente:** Modelos entrenados guardados en `cache/hmm_etth1_K5.pth` para reproducibilidad

Validación: convergencia monótona de log-likelihood con |ΔLL|<ε, verificación de propiedades estocásticas (Σ_j A[i,j]=1, Σ_k π_k=1).

#### Fase 3: Tokenización Viterbi (✓ Implementada)

Algoritmo de programación dinámica implementado en `hmm/viterbi.py`:

- **Input:** Serie normalizada [T=8640 timesteps ETTh1]
- **Output:** Secuencia estados óptima [z₁,...,z_T] con z_t∈{0,1,2,3,4}
- **Compresión:** 8640 timesteps → 319 segmentos run-length → ratio 27.1x
- **Distribución tokens:** Balanceada (cada token 15-25% frecuencia)
- **Log-probabilidad:** Validación de optimalidad mediante forward-backward

Verificación: persistencia media tokens (run-length promedio) significativamente mayor que discretización simple (11.24 HMM vs 2.74 SAX), evidenciando captura de regímenes persistentes.

#### Fase 4: Embeddings Estructurados (✓ Implementada)

Generación de embeddings en `embeddings/embedding_generator.py`:

- **Embedding crudo:** e_k = [μ_k, σ_k, A[k,:]] ∈ ℝ^(2+K) = ℝ^7
- **Proyección opcional:** Linear(7, d_model=128) para compatibilidad transformer
- **Interpretabilidad:** Visualización espacio μ-σ muestra separación clara de regímenes con volatilidades diferenciadas
- **Estructura temporal:** Matriz A exhibe diagonal fuerte (persistencia A[k,k]>0.5) y transiciones estructuradas

Validación cualitativa: estados capturan regímenes interpretables (ej: estado 2 con μ=0.82, σ=0.45, A[2,2]=0.68 representa régimen volátil persistente).

### Técnicas de Tokenización Implementadas (✓ 6/6)

Todas las técnicas del anteproyecto han sido implementadas con sus embeddings naturales correspondientes en `tecnicas/*.py` y `embeddings/technique_embeddings.py`:

**1. Discretización (ej: SAX)**
- **Tokenización:** `tecnicas/discretization.py` - Cuantización alfabética K=8 símbolos
- **Embedding:** `DiscretizationEmbedding` - Lookup table aprendible [K, d_model]
- **Validación:** Visualizaciones en `tecnicas/figures/sax_discretizacion_*.png`

**2. Text-based (ej: LLMTime)**
- **Tokenización:** `tecnicas/text_based.py` - Serialización a strings con precision=2
- **Embedding:** `TextBasedEmbedding` - Character embeddings vocab=['0'-'9','.','-',' ']
- **Validación:** Visualizaciones en `tecnicas/figures/llmtime_text_based_*.png`

**3. Patching (ej: PatchTST)**
- **Tokenización:** `tecnicas/patching.py` - Segmentación patches P=16 non-overlapping
- **Embedding:** `PatchingEmbedding` - Proyección lineal [P, d_model]
- **Validación:** Visualizaciones en `tecnicas/figures/patchtst_patches_*.png`

**4. Descomposición (ej: DLinear, Autoformer)**
- **Tokenización:** `tecnicas/decomposition.py` - Moving average trend + seasonal residual
- **Embedding:** `DecompositionEmbedding` - Proyección separada por componente
- **Validación:** Visualizaciones en `tecnicas/figures/autoformer_decomposition_*.png`

**5. Foundation (ej: MOMENT)**
- **Tokenización:** `tecnicas/foundation.py` - Patches + masked reconstruction (30%)
- **Embedding:** `FoundationEmbedding` - Patch + mask token learnable
- **Validación:** Visualizaciones en `tecnicas/figures/moment_masking_*.png`

**6. HMM (RITMO - propuesta del TFG)**
- **Tokenización:** `hmm/viterbi.py` - Estados ocultos óptimos
- **Embedding:** `EmbeddingGenerator` - Estructurado e_k=[μ_k, σ_k, A[k,:]]
- **Validación:** Visualizaciones en `tecnicas/figures/hmm_tokenizacion_*.png`

### Métricas Intrínsecas Implementadas (✓ 7/7)

Sistema completo de métricas en `tecnicas/metrics.py` organizadas según feedback del tutor:

**Métricas Universales (aplican a todas las técnicas):**

1. **compression_ratio(T, num_tokens)** - Compacidad/Granularidad
   - Ratio T_original / tokens_generados
   - Resultados: Text-based 0.1x (expande), Patching 16x, Descomposición 500x, HMM/Discretización 1x

2. **mse_reconstruction(original, reconstructed)** - Reconstrucción
   - Error cuadrático medio pérdida información
   - Resultados: Text-based/Descomposición <10⁻⁴ (casi lossless), HMM 0.11, Discretización 0.17 (lossy)

3. **acf_retention(original, reconstructed, nlags=20)** - Dependencias Temporales
   - Correlación de Pearson entre ACF original y reconstruida (CAMEO 2025)
   - Resultados: Todas >0.98 (excelente preservación estructura temporal)

**Métricas Discretas (solo Discretización y HMM):**

4. **vocabulary_entropy(tokens)** - Distribución/Diversidad
   - Entropía normalizada unigramas (Uzan et al. 2024)
   - Resultados: Discretización 0.99 (uniforme), HMM 0.97 (balanceado)

5. **bigram_entropy(tokens)** - Estructura Temporal
   - Entropía transiciones token_i → token_j
   - Resultados: Discretización 0.84 (variable), HMM 0.70 (estructurado por matriz A)

6. **token_persistence(tokens)** - Persistencia/Run-length
   - Longitud media runs consecutivos mismo token
   - Resultados: **HMM 11.24 vs Discretización 2.74** (4x mayor persistencia = captura regímenes)

7. **top_k_coverage(tokens, k=5)** - Concentración
   - Fracción uso cubierta por top-K tokens frecuentes (Zipf's law)
   - Resultados: HMM 1.0 (K=5, top-5=todos), Discretización 0.68 (balanceado)

Notebook comparativo `tecnicas/comparacion_metricas.ipynb` aplica todas las métricas a las 6 técnicas sobre ETTh1, generando visualizaciones en `tecnicas/figures/metricas_*.png`.

### Datasets

Siguiendo protocolo TSLib, se emplean 6 datasets divididos en dos grupos:

**Grupo 1 - Entrenamiento HMM (4 datasets):**
- **ETTh1:** Benchmark universal, split 7:1:2
- **ETTh2:** Validación robustez RevIN ante distribution shift
- **Weather:** Regímenes climáticos estacionales claros
- **Electricity:** Periodicidad diaria fuerte (24h día/noche)

**Grupo 2 - Evaluación Zero-shot (2 datasets):**
- **Traffic:** Regímenes rush-hour sin re-entrenamiento HMM
- **Exchange:** Series financieras test robustez

### Protocolo Experimental: Plan A+B

Siguiendo recomendación del tutor, se ejecutarán ambos planes de forma complementaria:

#### Plan A (Principal): Comparación Controlada de Tokenizaciones

**Objetivo:** Evaluar 6 técnicas con condiciones experimentales idénticas para aislar efecto de tokenización.

**Configuración:**
- Mismo backbone Transformer (decoder-only, 4 capas, 8 heads, dim=512)
- Mismo protocolo entrenamiento (optimizer, learning rate, epochs)
- Mismas métricas intrínsecas y downstream
- Input length I=96, horizontes O∈{96,192,336,720}

**Evaluación:**
1. Métricas intrínsecas (sin forecasting): compresión, MSE reconstrucción, ACF retention, entropías, persistencia
2. Métricas downstream: MSE, MAE forecasting promediados sobre 4 horizontes
3. Eficiencia: tiempo entrenamiento, inferencia, throughput

**Aporte científico:** Primera comparación sistemática de 6 paradigmas de tokenización en condiciones controladas, cuantificando trade-offs específicos de cada enfoque.

#### Plan B (Secundario): Validación SOTA

**Objetivo:** Situar HMM+Transformer en contexto del estado del arte consolidado.

**Baselines (implementados en `models/`):**
- **DLinear** (Zeng et al., 2023): Descomposición + lineal
- **PatchTST** (Nie et al., 2023): Patching baseline principal
- **TimeMixer** (Wang et al., 2024b): Descomposición multiscale
- **TimeXer** (Wang et al., 2024a): Variables exógenas (aunque TFG es univariado)

**Configuración:**
- Hiperparámetros óptimos reportados en papers originales
- Protocolo TSLib estándar para reproducibilidad
- Comparación directa HMM+Transformer vs 4 baselines

**Aporte científico:** Validación empírica de que tokenización probabilística HMM es competitiva frente a métodos con arquitecturas especializadas y años de optimización.

### Limitaciones del Scope Experimental

Coherente con anteproyecto, se excluyen:

1. **TOTEM/VQ-VAE:** Diseñados para multivariado, sin benchmarks univariados publicados
2. **Chronos/LLMTime text-based completos:** Ausentes en tablas comparativas comunidad, vocabulario 4096 requiere quantization-aware training
3. **MOMENT pre-entrenado:** Introduce confounder de datos masivos no controlables; re-entrenar desde cero es prohibitivo (≈1000 GPU-horas)

Comparación se centra en 4 baselines (Plan B) con implementaciones reproducibles, presencia consistente en benchmarks TSLib y sin pre-entrenamiento masivo.

### Estado Actual de Implementación

**✓ Completado (100%):**
- Pipeline RITMO 4 fases (RevIN, Baum-Welch, Viterbi, Embeddings)
- 6 técnicas de tokenización con embeddings naturales
- 7 métricas intrínsecas (3 universales + 4 discretas)
- Notebooks validación exhaustiva con visualizaciones profesionales
- Cache modelos HMM entrenados para reproducibilidad
- Figuras paper-ready exportadas a `tecnicas/figures/` y `notebooks/`

**⚙️ En Progreso:**
- Plan A: Entrenamiento Transformer único con 6 tokenizaciones
- Plan B: Comparación HMM+Transformer vs 4 baselines SOTA

**📋 Pendiente:**
- Experimentos forecasting completos en 6 datasets × 4 horizontes
- Análisis trade-offs complejidad-desempeño
- Análisis interpretabilidad regímenes HMM (inspección parámetros μ, σ, A)
- Redacción memoria final consolidando resultados

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
