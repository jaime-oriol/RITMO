"""
Encoder y Decoder estándar de Transformer para series temporales.
Implementación base usada por PatchTST y otros modelos.
"""

import torch  # Framework de deep learning
import torch.nn as nn  # Módulo de redes neuronales
import torch.nn.functional as F  # Funciones de activación


class ConvLayer(nn.Module):
    """
    Capa convolucional con downsampling.
    Reduce la secuencia a la mitad (para crear representaciones jerárquicas).
    """
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        # Convolución 1D con padding circular
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)  # Normalización
        self.activation = nn.ELU()  # Activación ELU
        # MaxPool reduce longitud a la mitad
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """x: [B, L, D] → [B, L/2, D]"""
        x = self.downConv(x.permute(0, 2, 1))  # [B, D, L]
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)  # Reduce longitud
        x = x.transpose(1, 2)  # [B, L/2, D]
        return x


class EncoderLayer(nn.Module):
    """
    Capa de encoder de Transformer estándar.
    Estructura: Self-Attention → Add&Norm → FFN → Add&Norm
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  # FFN típicamente 4x d_model

        self.attention = attention  # Mecanismo de atención
        # FFN: dos convoluciones 1x1 (equivalente a dos lineales)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # Normalizaciones
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        x: [B, L, d_model]
        Salida: [B, L, d_model], attention_weights
        """
        # === 1. SELF-ATTENTION + RESIDUAL ===
        new_x, attn = self.attention(
            x, x, x,  # Q, K, V todos iguales (self-attention)
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)  # Conexión residual

        # === 2. LAYER NORM ===
        y = x = self.norm1(x)

        # === 3. FFN + RESIDUAL ===
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """
    Encoder completo: apila múltiples EncoderLayers.
    Opcionalmente incluye capas convolucionales entre atenciones.
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer  # Normalización final

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        x: [B, L, D]
        Salida: [B, L', D], lista de attention weights
        """
        attns = []

        if self.conv_layers is not None:
            # Alternar atención y convolución (pirámide jerárquica)
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)  # Reduce secuencia
                attns.append(attn)
            # Última capa de atención sin conv
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            # Solo capas de atención
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        # Normalización final
        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Capa de decoder de Transformer.
    Estructura: Self-Attn → Cross-Attn → FFN (cada uno con Add&Norm)
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.self_attention = self_attention   # Atención sobre el decoder
        self.cross_attention = cross_attention  # Atención sobre el encoder
        # FFN
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # Tres normalizaciones
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        x: entrada del decoder [B, L_dec, d_model]
        cross: salida del encoder [B, L_enc, d_model]
        """
        # === 1. SELF-ATTENTION (masked) ===
        x = x + self.dropout(self.self_attention(
            x, x, x,  # Q, K, V del decoder
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        # === 2. CROSS-ATTENTION ===
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,  # Q del decoder, K, V del encoder
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        # === 3. FFN ===
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    """
    Decoder completo: apila múltiples DecoderLayers.
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection  # Proyección final a espacio de salida

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        x: entrada del decoder
        cross: salida del encoder
        """
        # Pasar por todas las capas
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        # Normalización final
        if self.norm is not None:
            x = self.norm(x)

        # Proyección final
        if self.projection is not None:
            x = self.projection(x)
        return x
