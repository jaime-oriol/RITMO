"""
TimeXer - Transformer con Variables Exógenas.
Idea: Usar variables externas (exógenas) para mejorar la predicción de una variable objetivo.
Separa embeddings endógenos (variable a predecir) y exógenos (variables auxiliares).
Paper: Wang et al., 2024
"""

import torch  # Framework de deep learning
import torch.nn as nn  # Módulo de redes neuronales
import torch.nn.functional as F  # Funciones de activación
from layers.SelfAttention_Family import FullAttention, AttentionLayer  # Mecanismo de atención
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding  # Embeddings


class FlattenHead(nn.Module):
    """
    Cabeza de predicción: aplana y proyecta a la ventana objetivo.
    Convierte la salida del encoder en predicciones finales.
    """
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars  # Número de variables
        self.flatten = nn.Flatten(start_dim=-2)  # Aplana las últimas 2 dims
        self.linear = nn.Linear(nf, target_window)  # Proyecta a ventana objetivo
        self.dropout = nn.Dropout(head_dropout)  # Regularización

    def forward(self, x):
        """x: [batch, nvars, d_model, patch_num] → [batch, nvars, target_window]"""
        x = self.flatten(x)  # Combina d_model y patch_num
        x = self.linear(x)   # Proyecta a longitud deseada
        x = self.dropout(x)  # Aplica dropout
        return x


class EnEmbedding(nn.Module):
    """
    Embedding para variable ENdógena (la que queremos predecir).
    Incluye un token GLOBAL que aprende información de toda la serie.
    """
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len  # Tamaño de cada patch

        # Proyecta cada patch a d_model dimensiones
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # Token global: aprende representación de toda la serie
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        # Codificación posicional: dónde está cada patch en la secuencia
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Divide en patches, añade token global, y aplica embeddings.
        x: [batch, nvars, seq_len] → [batch*nvars, num_patches+1, d_model]
        """
        n_vars = x.shape[1]  # Número de variables
        # Repetir token global para cada muestra del batch
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        # Patching: dividir serie en segmentos
        # unfold: crea patches no solapados de tamaño patch_len
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        # Reorganizar: [batch*nvars, num_patches, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # Embedding: valor + posicion
        x = self.value_embedding(x) + self.position_embedding(x)
        # Volver a separar por variable
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))

        # Anadir token global al final
        x = torch.cat([x, glb], dim=2)  # Concatenar en dimensión temporal

        # Aplanar para el encoder
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    """
    Encoder con cross-attention: procesa variable endógena usando info exógena.
    Similar al encoder de Transformer pero con atención cruzada.
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)  # Lista de capas del encoder
        self.norm = norm_layer  # Normalización final opcional
        self.projection = projection  # Proyección final opcional

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        x: embedding endógeno (variable a predecir)
        cross: embedding exógeno (variables auxiliares)
        """
        # Pasar por todas las capas del encoder
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        # Normalización final si existe
        if self.norm is not None:
            x = self.norm(x)

        # Proyección final si existe
        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    """
    Capa del encoder con DOBLE atención:
    1. Self-attention: la serie endógena se atiende a sí misma
    2. Cross-attention: el token global atiende a las variables exógenas
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  # Dimensión feedforward (4x d_model por defecto)

        # Dos mecanismos de atención
        self.self_attention = self_attention   # Para variable endógena
        self.cross_attention = cross_attention  # Para cruzar con exógenas

        # FFN: red feedforward (2 conv1d = 2 lineales)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        # 3 normalizaciones: una por cada sub-bloque
        self.norm1 = nn.LayerNorm(d_model)  # Después de self-attention
        self.norm2 = nn.LayerNorm(d_model)  # Después de cross-attention
        self.norm3 = nn.LayerNorm(d_model)  # Después de FFN

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        x: embedding endógeno [batch*nvars, num_patches+1, d_model]
        cross: embedding exógeno [batch, num_vars_exog, d_model]
        """
        B, L, D = cross.shape  # B=batch, L=vars exógenas, D=d_model

        # 1. Self-attention: la serie se atiende a si misma
        x = x + self.dropout(self.self_attention(
            x, x, x,  # Query, Key, Value = mismo x
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        # 2. Cross-attention: token global atiende a exogenas
        # Extraer solo el token global (último token)
        x_glb_ori = x[:, -1, :].unsqueeze(1)  # [batch*nvars, 1, d_model]
        # Reorganizar para cross-attention
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))  # [batch, nvars, d_model]

        # Cross-attention: global tokens atienden a variables exógenas
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,  # Q=global, K=V=exógenas
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        # Volver a forma original y aplicar conexión residual
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn  # Residual
        x_glb = self.norm2(x_glb)

        # Reemplazar token global actualizado en la secuencia
        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        # 3. Feedforward network
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)  # Residual + normalización


class Model(nn.Module):
    """
    TimeXer: Time Series with eXogenous variables.
    Usa variables externas para mejorar predicción de la variable objetivo.

    Arquitectura:
    1. Embedding endógeno (variable a predecir) con patches + token global
    2. Embedding exógeno (variables auxiliares) invertido
    3. Encoder con self-attention + cross-attention
    4. Cabeza de predicción
    """

    def __init__(self, configs):
        """
        Inicializa TimeXer.

        configs: Configuración del modelo (seq_len, pred_len, d_model, etc.)
        """
        super(Model, self).__init__()

        # Configuración básica
        self.task_name = configs.task_name
        self.features = configs.features  # 'M' multivariado, 'MS' multi-to-single
        self.seq_len = configs.seq_len  # Longitud de entrada
        self.pred_len = configs.pred_len  # Horizonte de predicción
        self.use_norm = configs.use_norm  # ¿Usar normalización RevIN?

        # Configuración de patching
        self.patch_len = configs.patch_len  # Tamaño de cada patch
        self.patch_num = int(configs.seq_len // configs.patch_len)  # Número de patches
        # Para MS: solo 1 variable endógena; para M: todas las variables
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in

        # Embeddings
        # Embedding endógeno: patches + token global
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        # Embedding exógeno: embedding invertido (timesteps como tokens)
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)

        # Encoder
        # Encoder con self-attention y cross-attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # Self-attention para variable endógena
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    # Cross-attention para cruzar con exógenas
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)  # Apilar e_layers capas
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Cabeza de prediccion
        # +1 por el token global
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Predicción con variables exógenas (modo MS: multi-to-single).
        Última variable es endógena, las demás son exógenas.

        x_enc: [batch, seq_len, channels]
        Salida: [batch, pred_len, 1]
        """
        if self.use_norm:
            # Normalizacion RevIN
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # N = número de variables

        # Separar endogena y exogenas
        # Última variable (-1) es la endógena (a predecir)
        # Resto (:-1) son exógenas (auxiliares)
        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        # Encoder: cross-attention entre endogena y exogenas
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [batch, nvars, d_model, patch_num]

        # Cabeza de prediccion
        dec_out = self.head(enc_out)  # [batch, nvars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [batch, pred_len, nvars]

        if self.use_norm:
            # Desnormalizacion
            # Solo desnormalizar la variable endógena (última)
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Predicción multivariada (modo M: todas las variables son endógenas).
        Cada variable se predice usando las demás como contexto.

        x_enc: [batch, seq_len, channels]
        Salida: [batch, pred_len, channels]
        """
        if self.use_norm:
            # Normalizacion RevIN
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        # Todas las variables son endogenas
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        # Encoder
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Cabeza de prediccion
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # Desnormalizacion
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Punto de entrada principal. Ejecuta según tipo de tarea y modo.

        x_enc: Serie de entrada [batch, seq_len, channels]
        x_mark_enc: Marcas temporales de entrada
        x_dec, x_mark_dec: No usados (encoder-only)
        mask: No usado en forecasting
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                # Modo multivariado: todas las variables son target
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]
            else:
                # Modo MS: última variable es target, resto son exógenas
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]
        else:
            return None
