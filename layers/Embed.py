"""
Módulo de Embeddings para series temporales.
Convierte datos crudos en representaciones vectoriales que los modelos pueden procesar.
Incluye: posicional, temporal, valor, y embeddings por patches.
"""

import torch  # Framework de deep learning
import torch.nn as nn  # Módulo de redes neuronales
import torch.nn.functional as F  # Funciones de activación
from torch.nn.utils import weight_norm  # Normalización de pesos
import math  # Funciones matemáticas


class PositionalEmbedding(nn.Module):
    """
    Embedding posicional sinusoidal (Vaswani et al., 2017).
    Codifica la posición de cada timestep usando senos y cosenos.
    No tiene parámetros aprendibles - es determinístico.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Crear matriz de embeddings posicionales [max_len, d_model]
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False  # No entrenable

        # Posiciones: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # Divisor para frecuencias: exp(2i * -log(10000)/d_model)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        # Posiciones pares: seno, impares: coseno
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # Registrar como buffer (no parámetro)

    def forward(self, x):
        """Devuelve embeddings posicionales para la longitud de x."""
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """
    Embedding de valores mediante convolución 1D.
    Proyecta cada timestep a d_model dimensiones usando una ventana de 3.
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # Conv1D: ventana de 3 timesteps, padding circular para bordes
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # Inicialización Kaiming para mejor convergencia
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """x: [B, T, C] → [B, T, d_model]"""
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """
    Embedding fijo (no entrenable) usando patrón sinusoidal.
    Similar a PositionalEmbedding pero como tabla de lookup.
    """
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        # Crear pesos sinusoidales fijos
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        # Usar nn.Embedding con pesos fijos
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """Lookup de embeddings fijos."""
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    Embedding temporal: codifica información de calendario.
    Suma embeddings de: mes, día, día de semana, hora, [minuto].
    """
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        # Tamaños de vocabulario para cada componente temporal
        minute_size = 4    # 0-3 (cuartos de hora)
        hour_size = 24     # 0-23
        weekday_size = 7   # 0-6 (lunes-domingo)
        day_size = 32      # 1-31
        month_size = 13    # 1-12

        # Elegir tipo de embedding: fijo o entrenable
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        # Crear embeddings para cada componente
        if freq == 't':  # Frecuencia por minutos
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        """
        x: [B, T, 5] con columnas [mes, día, día_semana, hora, minuto]
        Devuelve suma de todos los embeddings temporales.
        """
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """
    Embedding temporal mediante proyección lineal.
    Más simple que TemporalEmbedding: una sola capa lineal.
    """
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        # Dimensión de entrada según frecuencia
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        """Proyección lineal de features temporales."""
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    Embedding completo: valor + posición + temporal.
    Combina TokenEmbedding + PositionalEmbedding + TemporalEmbedding.
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # Elegir embedding temporal según tipo
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        x: valores [B, T, C]
        x_mark: marcas temporales [B, T, features] o None
        """
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    """
    Embedding invertido: trata variables como tokens (no timesteps).
    Usado en iTransformer y TimeXer para modelar relaciones entre variables.
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        # Proyección: toda la serie temporal → d_model
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        x: [B, T, C] → [B, C, d_model] (cada variable es un token)
        """
        x = x.permute(0, 2, 1)  # [B, C, T]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # Concatenar marcas temporales
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """
    Embedding sin posición explícita.
    Solo valor + temporal (la posición se aprende implícitamente).
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """Sin embedding posicional sumado."""
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Embedding por patches (PatchTST).
    Divide la serie en segmentos y proyecta cada uno a d_model.
    """
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len  # Longitud de cada patch
        self.stride = stride  # Paso entre patches (puede solapar)
        # Padding para que la última ventana no se pierda
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Proyección: patch → d_model
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # Embedding posicional para cada patch
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, C, T] → embeddings de patches [B*C, num_patches, d_model]
        """
        n_vars = x.shape[1]  # Número de variables

        # Añadir padding al final
        x = self.padding_patch_layer(x)
        # Extraer patches con unfold: [B, C, num_patches, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # Aplanar batch y variables: [B*C, num_patches, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # Proyectar + posición
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
