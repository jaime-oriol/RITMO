"""
TransformerCommon - Transformer encoder-only optimizado para Plan A.

Arquitectura FIJA diseñada específicamente para comparar técnicas de tokenización.
Acepta embeddings ya calculados de cualquier técnica y produce predicciones.

Características optimizadas:
- Encoder-only (suficiente para forecasting)
- Pre-norm architecture (más estable que post-norm)
- GELU activation (mejor que ReLU para series temporales)
- Adaptive pooling (maneja diferentes longitudes de secuencia)
- NO RevIN interno (se aplica externamente para comparación justa)
- Dropout estratégico (regularización sin overfitting)

Referencias:
    - Vaswani et al. 2017 (Transformer original)
    - Nie et al. 2023 (PatchTST: encoder-only para TS)
    - Kim et al. 2022 (RevIN se aplica externamente)
    - Xiong et al. 2020 (Pre-norm mejor que post-norm)
"""

import torch  # Framework de deep learning
import torch.nn as nn  # Módulo de redes neuronales
import math  # Operaciones matemáticas


class PositionalEncoding(nn.Module):
    """
    Positional encoding sinusoidal de Vaswani et al. 2017.

    Añade información de posición a los embeddings para que el Transformer
    sepa el orden temporal de la secuencia (transformers son permutation-invariant).

    Usa funciones seno/coseno de diferentes frecuencias:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Ventajas:
        - Determinístico (sin parámetros a entrenar)
        - Extrapola a secuencias más largas que las vistas en entrenamiento
        - Codifica distancias relativas (pos+k siempre tiene misma relación con pos)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Inicializa positional encoding.

        Args:
            d_model: Dimensión de los embeddings (debe ser par)
            max_len: Longitud máxima de secuencia soportada (default: 5000)
            dropout: Probabilidad de dropout después de añadir posición
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Crear matriz de positional encodings [max_len, d_model]
        # pe[pos, dim] = encoding de posición 'pos' en dimensión 'dim'
        pe = torch.zeros(max_len, d_model)

        # Posiciones: [0, 1, 2, ..., max_len-1] → shape [max_len, 1]
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Divisor para frecuencias: exp(-log(10000) * 2i/d_model)
        # Genera frecuencias desde rápidas (cambian cada pos) a lentas (cambian poco)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Aplicar seno a dimensiones pares, coseno a dimensiones impares
        pe[:, 0::2] = torch.sin(position * div_term)  # Columnas 0, 2, 4, ...
        pe[:, 1::2] = torch.cos(position * div_term)  # Columnas 1, 3, 5, ...

        # Añadir dimensión de batch: [1, max_len, d_model]
        # Esto permite broadcasting cuando se suma a [batch, seq_len, d_model]
        pe = pe.unsqueeze(0)

        # Registrar como buffer (no es parámetro entrenable, pero se guarda con el modelo)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Añade positional encoding a los embeddings.

        Args:
            x: Embeddings [batch, seq_len, d_model]

        Returns:
            Embeddings con información posicional [batch, seq_len, d_model]
        """
        # Sumar positional encoding (broadcasting automático en batch)
        # x + pe[:, :seq_len] suma el encoding correspondiente a cada posición
        x = x + self.pe[:, :x.size(1), :]

        # Aplicar dropout para regularización
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Capa de encoder con Pre-LN architecture (más estable que Post-LN).

    Estructura (Pre-Norm):
        1. LayerNorm → Multi-Head Attention → Residual
        2. LayerNorm → Feed-Forward → Residual

    Ventajas Pre-Norm sobre Post-Norm:
        - Más estable durante entrenamiento (gradientes más suaves)
        - Converge más rápido
        - No necesita learning rate warmup

    Referencias:
        - Xiong et al. 2020 "On Layer Normalization in the Transformer Architecture"
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Inicializa capa de encoder.

        Args:
            d_model: Dimensión de los embeddings (ej: 128)
            n_heads: Número de attention heads (ej: 4)
                    d_model debe ser divisible por n_heads
            d_ff: Dimensión de feed-forward network (típicamente 4*d_model)
            dropout: Probabilidad de dropout (default: 0.1)
        """
        super().__init__()

        # Verificar que d_model es divisible por n_heads
        assert d_model % n_heads == 0, f"d_model={d_model} debe ser divisible por n_heads={n_heads}"

        # === MULTI-HEAD SELF-ATTENTION ===
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True  # Formato [batch, seq, feature] (más intuitivo)
        )

        # === FEED-FORWARD NETWORK ===
        # Dos capas lineales con GELU en medio
        # GELU (Gaussian Error Linear Unit) funciona mejor que ReLU para series temporales
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),  # Expandir: 128 → 512
            nn.GELU(),                  # Activación suave (mejor que ReLU)
            nn.Dropout(dropout),        # Regularización
            nn.Linear(d_ff, d_model)    # Contraer: 512 → 128
        )

        # === LAYER NORMALIZATIONS (Pre-Norm) ===
        # Una antes de attention, otra antes de FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # === DROPOUT para conexiones residuales ===
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Procesa secuencia a través de la capa.

        Flujo Pre-Norm:
            x → Norm → Attention → Dropout → Add(x)
              → Norm → FFN → Dropout → Add

        Args:
            x: Entrada [batch, seq_len, d_model]
            src_mask: Máscara opcional [seq_len, seq_len]
                     True = posición enmascarada (no puede atender)

        Returns:
            Salida procesada [batch, seq_len, d_model]
        """
        # === 1. MULTI-HEAD SELF-ATTENTION ===
        # Pre-norm: normalizar ANTES de la operación
        x_norm = self.norm1(x)

        # Self-attention: cada posición atiende a todas las demás
        # attn_output tiene misma forma que entrada: [batch, seq_len, d_model]
        attn_output, _ = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=src_mask,
            need_weights=False  # No necesitamos los pesos (más rápido)
        )

        # Conexión residual: sumar entrada original (ayuda a flujo de gradientes)
        x = x + self.dropout1(attn_output)

        # === 2. FEED-FORWARD NETWORK ===
        # Pre-norm: normalizar ANTES de la operación
        x_norm = self.norm2(x)

        # Feed-forward: procesar cada posición independientemente
        ff_output = self.ff(x_norm)

        # Conexión residual: sumar entrada de este bloque
        x = x + self.dropout2(ff_output)

        return x


class Model(nn.Module):
    """
    TransformerCommon: Encoder-only Transformer para Plan A del TFG.

    Diseñado para comparación justa de técnicas de tokenización:
        - Arquitectura FIJA (misma para todas las técnicas)
        - Acepta embeddings ya calculados como entrada
        - NO hace RevIN interno (se aplica externamente para comparación justa)
        - Adaptive pooling para manejar longitudes variables

    Pipeline:
        Embeddings → Positional Encoding → Encoder Layers
                  → Adaptive Pool → Projection → Predicción

    Hiperparámetros optimizados para este caso:
        - d_model=128: Balance entre capacidad y eficiencia
        - n_heads=4: Suficiente para capturar dependencias múltiples
        - e_layers=2: Evita overfitting, suficiente para series temporales
        - d_ff=256: 2x d_model (más compacto que 4x tradicional)
        - dropout=0.1: Regularización moderada
    """

    def __init__(self, configs):
        """
        Inicializa el Transformer.

        Args:
            configs: Configuración con atributos:
                - seq_len: Longitud de entrada (varía por técnica)
                - pred_len: Horizonte de predicción (96, 192, 336, 720)
                - enc_in: Número de variables (1 para univariado)
                - d_model: Dimensión de embeddings (default: 128)
                - n_heads: Número de attention heads (default: 4)
                - e_layers: Número de capas del encoder (default: 2)
                - d_ff: Dimensión feed-forward (default: 256)
                - dropout: Probabilidad de dropout (default: 0.1)
        """
        super().__init__()

        # === GUARDAR CONFIGURACIÓN ===
        self.seq_len = configs.seq_len  # Longitud entrada (varía por técnica)
        self.pred_len = configs.pred_len  # Horizonte predicción
        self.d_model = getattr(configs, 'd_model', 128)  # Dimensión embeddings
        self.n_heads = getattr(configs, 'n_heads', 4)  # Attention heads
        self.e_layers = getattr(configs, 'e_layers', 2)  # Número de capas
        self.d_ff = getattr(configs, 'd_ff', 256)  # Dimensión FFN
        self.dropout = getattr(configs, 'dropout', 0.1)  # Dropout

        # === POSITIONAL ENCODING ===
        # Añade información de orden temporal a los embeddings
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model,
            max_len=5000,  # Soporta secuencias hasta 5000 tokens
            dropout=self.dropout
        )

        # === ENCODER LAYERS ===
        # Apilar e_layers capas de transformer
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout
            )
            for _ in range(self.e_layers)
        ])

        # === NORMALIZATION FINAL ===
        # Normalizar salida del encoder antes de projection
        self.norm = nn.LayerNorm(self.d_model)

        # === ADAPTIVE POOLING ===
        # Maneja diferentes longitudes de secuencia producidas por técnicas diferentes
        # Ejemplo: Discretización produce T=5000 tokens, Patching produce T=312 patches
        # Adaptive pooling las unifica a longitud fija antes de projection
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.pred_len)

        # === PROJECTION HEAD ===
        # Proyecta de d_model a la dimensión de salida (número de variables)
        self.projection = nn.Linear(self.d_model, configs.enc_in)

        # === INICIALIZACIÓN DE PESOS ===
        # Xavier initialization para mejor convergencia
        self._init_weights()

    def _init_weights(self):
        """
        Inicializa pesos del modelo para mejor convergencia.

        Xavier initialization: mantiene varianza constante entre capas
        (evita que activaciones exploten o se desvanezcan)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Linear layers: Xavier uniform
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # Biases: inicializar a cero
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm: peso=1, bias=0 (valores estándar)
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None, x_dec=None, x_mark_dec=None) -> torch.Tensor:
        """
        Forward pass del modelo.

        Pipeline completo:
            1. Añadir positional encoding
            2. Pasar por capas del encoder
            3. Pooling adaptativo (ajustar longitud)
            4. Projection a dimensión de salida

        NOTA: RevIN se aplica EXTERNAMENTE en exp_plan_a.py antes de tokenizar
              para garantizar comparación justa entre técnicas de tokenización.

        Args:
            x_enc: Embeddings de entrada [batch, seq_len, d_model]
                  Ya vienen embedidos por la técnica correspondiente
                  Los datos fueron normalizados ANTES de tokenizar
            x_mark_enc, x_dec, x_mark_dec: No usados (compatibilidad con API)

        Returns:
            Predicción normalizada [batch, pred_len, enc_in]
            (debe desnormalizarse externamente con RevIN en exp_plan_a.py)
        """
        # === 1. POSITIONAL ENCODING ===
        # Añadir información de posición a los embeddings
        # x_enc ya viene de datos normalizados (RevIN aplicado externamente)
        # x_enc: [batch, seq_len, d_model] → [batch, seq_len, d_model]
        x = self.pos_encoder(x_enc)

        # === 2. ENCODER LAYERS ===
        # Pasar secuencia por todas las capas del encoder
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)  # [batch, seq_len, d_model] → [batch, seq_len, d_model]

        # === 3. NORMALIZATION FINAL ===
        # Normalizar salida del encoder
        x = self.norm(x)  # [batch, seq_len, d_model]

        # === 4. ADAPTIVE POOLING ===
        # Reducir/expandir longitud de secuencia a pred_len
        # Diferentes técnicas producen diferentes seq_len:
        #   - Discretización: seq_len = 5000
        #   - Patching: seq_len = 312
        # Adaptive pooling unifica a pred_len (ej: 96, 192, 336, 720)

        # Transponer para pooling: [batch, seq_len, d_model] → [batch, d_model, seq_len]
        x = x.transpose(1, 2)

        # Pooling adaptativo: [batch, d_model, seq_len] → [batch, d_model, pred_len]
        x = self.adaptive_pool(x)

        # Transponer de vuelta: [batch, d_model, pred_len] → [batch, pred_len, d_model]
        x = x.transpose(1, 2)

        # === 5. PROJECTION HEAD ===
        # Proyectar de d_model a número de variables de salida
        # [batch, pred_len, d_model] → [batch, pred_len, enc_in]
        x = self.projection(x)

        # Salida está en espacio normalizado
        # Debe desnormalizarse EXTERNAMENTE en exp_plan_a.py con RevIN
        return x

    def forecast(self, x_enc: torch.Tensor, x_mark_enc=None, x_dec=None, x_mark_dec=None) -> torch.Tensor:
        """
        Alias para forward() - compatibilidad con API de exp/.

        Args:
            x_enc: Embeddings [batch, seq_len, d_model]

        Returns:
            Predicción [batch, pred_len, enc_in]
        """
        return self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
