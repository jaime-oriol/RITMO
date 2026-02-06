"""
Embeddings naturales para cada técnica de tokenización.

¿Qué es un embedding?
    Un embedding convierte algo discreto (un símbolo, un token) en un vector
    de números que una red neuronal puede procesar. Es como traducir palabras
    a coordenadas en un espacio donde palabras similares están cerca.

¿Por qué cada técnica tiene su propio embedding?
    Porque cada técnica produce tokens diferentes:
    - SAX produce símbolos discretos (a, b, c...)
    - PatchTST produce vectores continuos (patches de 16 valores)
    - LLMTime produce texto ("2 3 . 4 5")

    Usar el mismo embedding para todas sería como usar el mismo traductor
    para español, chino y código morse. No tiene sentido.

Referencias:
    - SAX: Lin et al. 2007 (one-hot o learnable)
    - LLMTime: Gruver et al. 2023 (character embeddings)
    - PatchTST: Nie et al. 2023 (linear projection)
    - Decomp: Zeng et al. 2023 (separate linear layers)
    - MOMENT: Goswami et al. 2024 (patch + position + mask embeddings)
"""

import torch  # Framework de deep learning
import torch.nn as nn  # Módulo de redes neuronales
import numpy as np  # Operaciones con arrays
from typing import Dict, Optional  # Para tipos de datos


class DiscretizationEmbedding(nn.Module):
    """
    Embedding para técnica de DISCRETIZACIÓN: tabla de embeddings aprendibles.

    La discretización (ej: SAX, VQ-VAE, TOTEM) convierte cada valor de la serie
    en un símbolo discreto de un vocabulario finito.
    Internamente usamos números: símbolo 0, 1, 2, ..., vocab_size-1.

    El embedding es una TABLA donde cada fila corresponde a un símbolo:
        Símbolo 0 → [0.23, -0.45, 0.12, ...]  (vector de d_model números)
        Símbolo 1 → [-0.11, 0.67, -0.33, ...]
        ...

    Cuando la red ve el símbolo 3, busca la fila 3 en la tabla y devuelve ese vector.
    Esta tabla se APRENDE durante el entrenamiento (los números se ajustan para
    que símbolos similares tengan vectores parecidos).

    Entrada: tokens [T] con valores 0, 1, 2, ..., vocab_size-1
    Salida: embeddings [T, d_model] - un vector por cada token
    """

    def __init__(self, vocab_size: int = 8, d_model: int = 128):
        """
        Inicializa la tabla de embeddings.

        Args:
            vocab_size: Tamaño del vocabulario discreto (default: 8)
                       Número de símbolos únicos que produce la discretización
            d_model: Tamaño del vector de salida (default: 128)
                    Cada símbolo se convierte en un vector de 128 números
        """
        # Llamar al constructor de nn.Module (obligatorio en PyTorch)
        super().__init__()

        # Guardar configuración
        self.vocab_size = vocab_size  # Cuántos símbolos
        self.d_model = d_model  # Tamaño del vector: 128

        # CREAR LA TABLA DE EMBEDDINGS
        # Es una matriz de [vocab_size filas x d_model columnas]
        # Cada fila = embedding de un símbolo
        # Los valores iniciales son aleatorios, se aprenden durante entrenamiento
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Inicialización Xavier: pone valores iniciales "buenos" para que
        # el entrenamiento sea más estable
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convierte tokens discretos en embeddings (la operación principal).

        Ejemplo:
            tokens = [0, 3, 7, 2]  (4 símbolos discretos)
            embeddings = [
                [0.23, -0.45, ...],   # vector del símbolo 0
                [0.56, 0.12, ...],    # vector del símbolo 3
                [0.89, 0.22, ...],    # vector del símbolo 7
                [-0.33, 0.44, ...]    # vector del símbolo 2
            ]

        Args:
            tokens: Tensor con índices de símbolos
                   Puede ser [T] para una serie
                   o [B, T] para un batch de B series

        Returns:
            Embeddings [T, d_model] o [B, T, d_model]
        """
        # nn.Embedding hace la búsqueda automáticamente:
        # para cada número en 'tokens', busca esa fila en la tabla
        return self.embedding(tokens)

    def from_numpy(self, tokens: np.ndarray) -> torch.Tensor:
        """
        Versión que acepta arrays de numpy (más cómodo para usar).

        Convierte el array numpy a tensor PyTorch y llama a forward().
        """
        # Convertir numpy array a tensor de enteros largos (long = int64)
        tokens_tensor = torch.from_numpy(tokens).long()
        return self.forward(tokens_tensor)


class TextBasedEmbedding(nn.Module):
    """
    Embedding para LLMTime: embeddings a nivel de caracter.

    LLMTime convierte cada valor numérico en texto separando dígitos:
        23.45 → " 2 3 . 4 5"
        -12.30 → "- 1 2 . 3 0"

    El embedding asigna un vector a CADA CARACTER:
        '2' → [0.12, -0.33, ...]
        '3' → [0.45, 0.22, ...]
        '.' → [-0.11, 0.67, ...]

    Luego, para obtener UN embedding por timestep, promediamos todos los
    caracteres de ese valor:
        " 2 3 . 4 5" tiene 9 caracteres → promedio de 9 vectores → 1 vector

    El paper original de LLMTime usa el tokenizer de GPT (muy complejo).
    Aquí usamos character embeddings aprendibles para comparación justa.

    Entrada: lista de strings, un string por timestep
    Salida: embeddings [T, d_model] - un vector por timestep
    """

    # VOCABULARIO: qué caracteres reconocemos y su número asignado
    # Son solo 14 caracteres posibles en números decimales
    VOCAB = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,  # Dígitos 0-4
        '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,  # Dígitos 5-9
        '.': 10,     # Punto decimal
        '-': 11,     # Signo negativo
        ' ': 12,     # Espacio (separador)
        '<pad>': 13  # Caracter especial para relleno
    }

    def __init__(self, d_model: int = 128):
        """
        Inicializa embeddings de caracteres.

        Args:
            d_model: Tamaño del vector de salida (default: 128)
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = len(self.VOCAB)  # 14 caracteres

        # TABLA DE EMBEDDINGS PARA CARACTERES
        # Matriz [14 filas x 128 columnas]
        # Fila 0 = embedding de '0'
        # Fila 10 = embedding de '.'
        # etc.
        self.char_embedding = nn.Embedding(self.vocab_size, d_model)
        nn.init.xavier_uniform_(self.char_embedding.weight)

    def _tokenize_string(self, text: str) -> torch.Tensor:
        """
        Convierte un string a tensor de índices.

        Ejemplo:
            " 2 3" → [12, 2, 12, 3]  (espacio=12, '2'=2, espacio=12, '3'=3)

        Args:
            text: String a convertir

        Returns:
            Tensor con índices de cada caracter
        """
        # Para cada caracter, buscar su índice en VOCAB
        # Si no existe, usar '<pad>' (índice 13)
        indices = [self.VOCAB.get(c, self.VOCAB['<pad>']) for c in text]
        return torch.tensor(indices, dtype=torch.long)

    def forward(self, tokens_per_value: list) -> torch.Tensor:
        """
        Convierte lista de strings en embeddings.

        Proceso:
            1. Cada string se convierte a índices de caracteres
            2. Cada índice se busca en la tabla de embeddings
            3. Se promedian todos los vectores del string → 1 vector por timestep

        Ejemplo:
            tokens_per_value = [" 2 3 . 4 5", "- 1 2 . 3 0"]
            → 2 vectores de [d_model] dimensiones

        Args:
            tokens_per_value: Lista de T strings
                             Cada string representa un valor serializado
                             Viene de text_based_tokenize()['tokens_per_value']

        Returns:
            Embeddings [T, d_model]
        """
        T = len(tokens_per_value)  # Número de timesteps
        embeddings = []

        for value_str in tokens_per_value:
            # Paso 1: Convertir string a índices
            # " 2 3" → [12, 2, 12, 3]
            char_indices = self._tokenize_string(value_str)

            # Paso 2: Buscar cada índice en la tabla de embeddings
            # [12, 2, 12, 3] → [[vec_espacio], [vec_2], [vec_espacio], [vec_3]]
            char_embs = self.char_embedding(char_indices)  # [num_chars, d_model]

            # Paso 3: Promediar todos los vectores → 1 vector por timestep
            # Esto "resume" toda la información del número en un solo vector
            value_emb = char_embs.mean(dim=0)  # [d_model]
            embeddings.append(value_emb)

        # Apilar todos los embeddings en un tensor [T, d_model]
        return torch.stack(embeddings)


class PatchingEmbedding(nn.Module):
    """
    Embedding para técnica de PATCHING: proyección lineal de patches.

    El patching (ej: PatchTST, Crossformer) divide la serie en "parches" de longitud fija:
        Serie [0, 1, 2, ..., 99] con patch_len=16
        → Patch 0: [0, 1, 2, ..., 15]
        → Patch 1: [8, 9, 10, ..., 23]  (si stride=8)
        → ...

    El embedding es una PROYECCIÓN LINEAL:
        embedding = W @ patch + b

    Donde W es una matriz [d_model x patch_len] que se aprende.
    Esto es como comprimir los 16 valores del patch en 128 valores
    que capturan la "esencia" del patch.

    También añadimos POSITIONAL EMBEDDING: un vector que indica la posición
    del patch en la secuencia (patch 0, patch 1, ...). Esto ayuda al
    transformer a saber el orden temporal.

    Entrada: patches [num_patches, patch_len] - matriz de parches
    Salida: embeddings [num_patches, d_model] - un vector por patch
    """

    def __init__(self, patch_len: int = 16, d_model: int = 128):
        """
        Inicializa proyección lineal y positional embeddings.

        Args:
            patch_len: Longitud de cada patch (default: 16)
                      Cada patch tiene 16 valores de la serie original
            d_model: Tamaño del vector de salida (default: 128)
        """
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model

        # PROYECCIÓN LINEAL
        # Transforma vector de 16 valores → vector de 128 valores
        # Los pesos W y bias b se aprenden durante entrenamiento
        self.projection = nn.Linear(patch_len, d_model)

        # POSITIONAL EMBEDDING
        # Tabla donde cada fila indica "eres el patch número X"
        # Esto es importante porque transformers no tienen noción de orden
        self.max_patches = 512  # Máximo número de patches que soportamos
        self.position_embedding = nn.Embedding(self.max_patches, d_model)

    def forward(self, patches: torch.Tensor, add_position: bool = True) -> torch.Tensor:
        """
        Proyecta patches a embeddings.

        Proceso:
            1. Multiplicar cada patch por matriz W: [16] → [128]
            2. (Opcional) Sumar positional embedding: indica posición temporal

        Args:
            patches: Tensor [num_patches, patch_len]
                    Cada fila es un patch de 16 valores
            add_position: Si añadir información de posición (default: True)

        Returns:
            Embeddings [num_patches, d_model]
        """
        # Paso 1: Proyección lineal
        # Cada patch [16] → embedding [128]
        embeddings = self.projection(patches)

        # Paso 2: Añadir positional embedding
        if add_position:
            if patches.dim() == 2:
                # Caso: un solo batch de patches [num_patches, patch_len]
                num_patches = patches.shape[0]
                # Crear tensor [0, 1, 2, ..., num_patches-1]
                positions = torch.arange(num_patches, device=patches.device)
                # Buscar embedding de cada posición y sumar
                embeddings = embeddings + self.position_embedding(positions)
            else:
                # Caso: batch de batches [B, num_patches, patch_len]
                num_patches = patches.shape[1]
                positions = torch.arange(num_patches, device=patches.device)
                embeddings = embeddings + self.position_embedding(positions).unsqueeze(0)

        return embeddings

    def from_numpy(self, patches: np.ndarray, add_position: bool = True) -> torch.Tensor:
        """Versión que acepta arrays de numpy."""
        patches_tensor = torch.from_numpy(patches).float()
        return self.forward(patches_tensor, add_position)


class DecompositionEmbedding(nn.Module):
    """
    Embedding para técnica de DESCOMPOSICIÓN: proyección por componente.

    La descomposición (ej: DLinear, Autoformer, FEDformer) separa la serie en dos partes:
        - Trend: tendencia suave (¿la serie sube o baja en general?)
        - Seasonal: variación cíclica (patrones que se repiten)

    DLinear trata cada componente POR SEPARADO con proyecciones lineales
    diferentes, y luego los combina.

    Esto tiene sentido porque:
        - El trend es lento y suave → necesita aprender patrones de largo plazo
        - El seasonal es rápido y cíclico → necesita aprender periodicidades

    Entrada: trend [T] y seasonal [T] - dos series de igual longitud
    Salida: embeddings [T, d_model] - combinación de ambos componentes
    """

    def __init__(self, d_model: int = 128):
        """
        Inicializa proyecciones para trend y seasonal.

        Args:
            d_model: Tamaño del vector de salida (default: 128)
                    La mitad (64) viene del trend, la otra mitad del seasonal
        """
        super().__init__()
        self.d_model = d_model

        # PROYECCIÓN PARA TREND
        # Cada valor escalar del trend → vector de d_model/2 = 64
        self.trend_proj = nn.Linear(1, d_model // 2)

        # PROYECCIÓN PARA SEASONAL
        # Cada valor escalar del seasonal → vector de d_model/2 = 64
        self.seasonal_proj = nn.Linear(1, d_model // 2)

        # LAYER NORM: normaliza los valores para que estén en rango estable
        # Esto ayuda al entrenamiento
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trend: torch.Tensor, seasonal: torch.Tensor) -> torch.Tensor:
        """
        Combina trend y seasonal en embeddings.

        Proceso:
            1. Proyectar trend: valor → vector de 64
            2. Proyectar seasonal: valor → vector de 64
            3. Concatenar: [64] + [64] = [128]
            4. Normalizar para estabilidad

        Args:
            trend: Serie de tendencia [T]
            seasonal: Serie estacional [T]

        Returns:
            Embeddings [T, d_model]
        """
        # Asegurar que los tensores tienen forma [T, 1] para la proyección lineal
        # Linear espera entrada [..., 1] para producir [..., d_model/2]
        if trend.dim() == 1:
            trend = trend.unsqueeze(-1)  # [T] → [T, 1]
            seasonal = seasonal.unsqueeze(-1)  # [T] → [T, 1]
        elif trend.dim() == 2 and trend.shape[-1] != 1:
            # Caso [B, T] → [B, T, 1]
            trend = trend.unsqueeze(-1)
            seasonal = seasonal.unsqueeze(-1)

        # Paso 1: Proyectar trend
        # [T, 1] → [T, 64]
        trend_emb = self.trend_proj(trend)

        # Paso 2: Proyectar seasonal
        # [T, 1] → [T, 64]
        seasonal_emb = self.seasonal_proj(seasonal)

        # Paso 3: Concatenar
        # [T, 64] + [T, 64] → [T, 128]
        embeddings = torch.cat([trend_emb, seasonal_emb], dim=-1)

        # Paso 4: Normalizar
        embeddings = self.norm(embeddings)

        return embeddings

    def from_numpy(self, trend: np.ndarray, seasonal: np.ndarray) -> torch.Tensor:
        """Versión que acepta arrays de numpy."""
        trend_tensor = torch.from_numpy(trend).float()
        seasonal_tensor = torch.from_numpy(seasonal).float()
        return self.forward(trend_tensor, seasonal_tensor)


class FoundationEmbedding(nn.Module):
    """
    Embedding para técnica de FOUNDATION MODELS: patch embedding + position + mask token.

    Los foundation models (ej: MOMENT, Timer, MOIRAI) usan pre-entrenamiento masivo
    con la tarea de reconstruir patches enmascarados (como BERT pero para series temporales).

    El embedding combina:
        1. Proyección lineal de patches (igual que PatchTST)
        2. Positional embedding (indica posición del patch)
        3. Token [MASK] especial para patches ocultos

    Durante pre-training, algunos patches se reemplazan con [MASK] y el modelo
    debe predecir sus valores originales. Esto le enseña a entender patrones.

    Entrada: patches [num_patches, patch_len] y máscara [num_patches]
    Salida: embeddings [num_patches, d_model]
    """

    def __init__(self, patch_len: int = 16, d_model: int = 128):
        """
        Inicializa componentes del embedding.

        Args:
            patch_len: Longitud de cada patch (default: 16)
            d_model: Tamaño del vector de salida (default: 128)
        """
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model

        # PROYECCIÓN DE PATCHES (igual que PatchTST)
        self.patch_projection = nn.Linear(patch_len, d_model)

        # TOKEN [MASK]
        # Es un vector aprendible que reemplaza a los patches enmascarados
        # El modelo aprende qué valores poner aquí para indicar "esto está oculto"
        self.mask_token = nn.Parameter(torch.randn(1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)  # Inicialización pequeña

        # POSITIONAL EMBEDDING
        self.max_patches = 512
        self.position_embedding = nn.Embedding(self.max_patches, d_model)

        # LAYER NORM
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patches: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Genera embeddings aplicando masking si se proporciona.

        Proceso:
            1. Proyectar todos los patches: [16] → [128]
            2. Para patches con mask=True, reemplazar con [MASK] token
            3. Añadir positional embedding
            4. Normalizar

        Ejemplo:
            patches = [[1,2,3,...,16], [17,18,...,32], [33,34,...,48]]
            mask = [False, True, False]
            → El patch 1 se reemplaza por [MASK], los otros se proyectan normal

        Args:
            patches: Tensor [num_patches, patch_len]
            mask: Tensor booleano [num_patches]
                  True = este patch está enmascarado (oculto)
                  None = no aplicar masking

        Returns:
            Embeddings [num_patches, d_model]
        """
        num_patches = patches.shape[0]

        # Paso 1: Proyectar todos los patches
        embeddings = self.patch_projection(patches)  # [num_patches, d_model]

        # Paso 2: Reemplazar patches enmascarados con [MASK] token
        if mask is not None:
            # Expandir máscara para que tenga misma forma que embeddings
            # mask [num_patches] → [num_patches, d_model]
            mask_expanded = mask.unsqueeze(-1).expand_as(embeddings)

            # Crear tensor de [MASK] tokens para todos los patches
            mask_tokens = self.mask_token.expand(num_patches, -1)

            # torch.where: si mask=True, usar mask_token; si no, usar embedding original
            embeddings = torch.where(mask_expanded, mask_tokens, embeddings)

        # Paso 3: Añadir positional embedding
        positions = torch.arange(num_patches, device=patches.device)
        embeddings = embeddings + self.position_embedding(positions)

        # Paso 4: Normalizar
        embeddings = self.norm(embeddings)

        return embeddings

    def from_numpy(self, patches: np.ndarray, mask: np.ndarray = None) -> torch.Tensor:
        """Versión que acepta arrays de numpy."""
        patches_tensor = torch.from_numpy(patches).float()
        mask_tensor = torch.from_numpy(mask).bool() if mask is not None else None
        return self.forward(patches_tensor, mask_tensor)


# Diccionario para acceso por nombre de tecnica
# Permite obtener la clase de embedding usando el nombre de la TÉCNICA.
# Los nombres corresponden a las 5 técnicas del anteproyecto, NO a modelos específicos.

EMBEDDINGS = {
    # Técnica: Discretización (ej: SAX, VQ-VAE, TOTEM, Chronos)
    'discretization': DiscretizationEmbedding,
    'discretizacion': DiscretizationEmbedding,  # Alias español

    # Técnica: Text-based (ej: LLMTime, Time-LLM)
    'text_based': TextBasedEmbedding,
    'text-based': TextBasedEmbedding,  # Alias con guión

    # Técnica: Patching (ej: PatchTST, Crossformer)
    'patching': PatchingEmbedding,

    # Técnica: Descomposición (ej: DLinear, Autoformer, FEDformer)
    'decomposition': DecompositionEmbedding,
    'descomposicion': DecompositionEmbedding,  # Alias español

    # Técnica: Foundation Models (ej: MOMENT, Timer, MOIRAI)
    'foundation': FoundationEmbedding,
}


def get_embedding(technique: str, **kwargs) -> nn.Module:
    """
    Factory: obtiene el embedding correcto dado el nombre de la TÉCNICA.

    IMPORTANTE: Usar nombres de TÉCNICAS, no de modelos específicos.

    Args:
        technique: Nombre de la técnica (según anteproyecto). Opciones:
                  - 'discretization': para técnica de discretización
                  - 'text_based': para técnica text-based
                  - 'patching': para técnica de patching
                  - 'decomposition': para técnica de descomposición
                  - 'foundation': para técnica de foundation models

        **kwargs: Parámetros específicos del embedding:
                 - d_model: dimensión de salida (todos)
                 - vocab_size: para discretization
                 - patch_len: para patching y foundation

    Returns:
        Instancia del embedding correspondiente, lista para usar

    Ejemplo:
        >>> emb = get_embedding('discretization', vocab_size=8, d_model=128)
        >>> tokens = np.array([0, 3, 7, 2, 1])
        >>> embeddings = emb.from_numpy(tokens)  # [5, 128]

        >>> emb = get_embedding('patching', patch_len=16, d_model=256)
        >>> patches = np.random.randn(10, 16)  # 10 patches de 16 valores
        >>> embeddings = emb.from_numpy(patches)  # [10, 256]
    """
    # Convertir a minúsculas para evitar errores por mayúsculas
    technique = technique.lower()

    # Verificar que la técnica existe
    if technique not in EMBEDDINGS:
        raise ValueError(f"Técnica '{technique}' no reconocida. "
                        f"Opciones válidas: {list(EMBEDDINGS.keys())}")

    # Crear y devolver instancia del embedding
    return EMBEDDINGS[technique](**kwargs)
