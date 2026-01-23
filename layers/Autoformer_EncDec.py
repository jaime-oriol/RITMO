"""
Componentes de Autoformer: Encoder/Decoder con descomposición progresiva.
Idea clave: En cada capa, separar la serie en tendencia y estacionalidad.
Esto permite que el modelo capture patrones a diferentes escalas temporales.
"""

import torch  # Framework de deep learning
import torch.nn as nn  # Módulo de redes neuronales
import torch.nn.functional as F  # Funciones de activación


class my_Layernorm(nn.Module):
    """
    LayerNorm especial para componente estacional.
    Normaliza y luego resta la media para centrar en cero.
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)  # Normalización estándar
        # Restar media temporal para centrar la estacionalidad en cero
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Media móvil para extraer la tendencia de una serie temporal.
    Suaviza la serie promediando ventanas de tamaño kernel_size.
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        # AvgPool1d: promedia ventanas deslizantes
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding simétrico: replicar extremos para mantener longitud
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)  # Inicio
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)    # Final
        x = torch.cat([front, x, end], dim=1)  # Concatenar padding

        # Aplicar promedio móvil (requiere formato [B, C, T])
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)  # Volver a [B, T, C]
        return x


class series_decomp(nn.Module):
    """
    Descomposición de series: separa tendencia y estacionalidad.
    Tendencia = media móvil (componente suave)
    Estacionalidad = original - tendencia (componente fluctuante)
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)  # Tendencia = promedio móvil
        res = x - moving_mean  # Estacionalidad = residuo
        return res, moving_mean  # (estacional, tendencia)


class series_decomp_multi(nn.Module):
    """
    Descomposición múltiple: usa varios tamaños de ventana y promedia.
    Idea de FEDformer: capturar patrones a diferentes escalas.
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        # Crear un descomponedor por cada tamaño de kernel
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        # Aplicar cada descomposición
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        # Promediar todos los resultados
        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class EncoderLayer(nn.Module):
    """
    Capa de encoder de Autoformer con descomposición progresiva.
    Flujo: Attention → Descomp → FFN → Descomp
    En cada paso, separa y descarta la tendencia, solo pasa la estacionalidad.
    """

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  # Dimensión feedforward

        self.attention = attention  # Mecanismo de atención
        # FFN como 2 convoluciones 1x1 (equivalente a 2 lineales)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        # Dos bloques de descomposición
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # === 1. SELF-ATTENTION ===
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)  # Conexión residual

        # === 2. PRIMERA DESCOMPOSICIÓN ===
        x, _ = self.decomp1(x)  # Descartar tendencia, quedarse con estacional

        # === 3. FEEDFORWARD NETWORK ===
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # === 4. SEGUNDA DESCOMPOSICIÓN ===
        res, _ = self.decomp2(x + y)  # Descartar tendencia
        return res, attn


class Encoder(nn.Module):
    """
    Encoder completo de Autoformer.
    Apila múltiples EncoderLayers con descomposición progresiva.
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []

        if self.conv_layers is not None:
            # Con capas convolucionales intermedias
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            # Solo capas de atención
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        # Normalización final
        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Capa de decoder de Autoformer con descomposición progresiva.
    Flujo: Self-Attn → Decomp → Cross-Attn → Decomp → FFN → Decomp
    Acumula las tendencias extraídas para la predicción final.
    """

    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.self_attention = self_attention
        self.cross_attention = cross_attention
        # FFN
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        # Tres bloques de descomposición
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        # Proyección para las tendencias acumuladas
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # === 1. SELF-ATTENTION + DESCOMPOSICIÓN ===
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)

        # === 2. CROSS-ATTENTION + DESCOMPOSICIÓN ===
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)

        # === 3. FFN + DESCOMPOSICIÓN ===
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        # === 4. ACUMULAR TENDENCIAS ===
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Decoder completo de Autoformer.
    Acumula las tendencias de cada capa para la predicción final.
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        # Pasar por cada capa acumulando tendencias
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend  # Acumular tendencia

        # Normalización y proyección final
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)

        return x, trend
