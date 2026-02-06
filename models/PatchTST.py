"""
PatchTST - Transformer con Patches para series temporales.
Idea: Dividir la serie en "patches" (segmentos) y tratarlos como tokens en un Transformer.
Reduce complejidad de O(L²) a O((L/P)²) donde P es tamaño del patch.
Paper: https://arxiv.org/pdf/2211.14730.pdf (Nie et al., 2023)
"""

import torch  # Framework de deep learning
from torch import nn  # Módulo de redes neuronales
from layers.Transformer_EncDec import Encoder, EncoderLayer  # Encoder Transformer
from layers.SelfAttention_Family import FullAttention, AttentionLayer  # Mecanismo de atención
from layers.Embed import PatchEmbedding  # Convierte patches en embeddings


class Transpose(nn.Module):
    """Utilidad para transponer dimensiones de tensores."""
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


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


class Model(nn.Module):
    """
    PatchTST: Patch-based Time Series Transformer.
    Divide la serie en patches, los procesa con un Transformer, y predice el futuro.
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        Inicializa PatchTST.

        configs: Configuración del modelo
        patch_len: Tamaño de cada patch (16 = cada patch tiene 16 timesteps)
        stride: Paso entre patches (8 = patches solapados a la mitad)
        """
        super().__init__()

        # Configuración básica
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # Capa de Patching + Embedding
        # Convierte serie [B, T, C] en patches [B*C, num_patches, d_model]
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder Transformer
        # Procesa los patches con self-attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)  # Apilar e_layers capas
            ],
            # Normalización: BatchNorm en la dimensión del modelo
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )

        # Calcular dimensión de salida del encoder
        # num_patches = (seq_len - patch_len) / stride + 2 (con padding)
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)

        # Cabeza de predicción según la tarea
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Predicción: proyectar a pred_len timesteps
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            # Reconstrucción: proyectar a seq_len timesteps
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            # Clasificación: aplanar y proyectar a clases
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Predicción de series temporales.

        x_enc: Entrada [batch, seq_len, channels]
        Salida: Predicción [batch, pred_len, channels]
        """
        # Normalizacion (RevIN-style)
        # Restar media y dividir por desviación estándar
        means = x_enc.mean(1, keepdim=True).detach()  # Media por serie
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Patching y embedding
        # Cambiar a [batch, channels, seq_len] para patching
        x_enc = x_enc.permute(0, 2, 1)
        # Crear patches: [batch*channels, num_patches, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder Transformer
        # Procesar patches con self-attention
        enc_out, attns = self.encoder(enc_out)
        # Reorganizar: [batch, channels, num_patches, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # Transponer: [batch, channels, d_model, num_patches]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Cabeza de prediccion
        dec_out = self.head(enc_out)  # [batch, channels, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [batch, pred_len, channels]

        # Desnormalizacion
        # Revertir la normalización inicial
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        Rellenar valores faltantes en la serie.
        Similar a forecast pero normaliza solo con valores observados (mask=1).
        """
        # Normalización considerando solo valores observados
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)  # Poner 0 donde falta
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # Patching y Encoder (igual que forecast)
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Cabeza de reconstrucción
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # Desnormalización
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        """
        Detección de anomalías: reconstruir la serie y comparar con original.
        Anomalías = puntos donde la reconstrucción difiere mucho.
        """
        # Normalización estándar
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Patching y Encoder
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Reconstrucción
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # Desnormalización
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        Clasificar la serie temporal en una categoría.
        Usa la representación del encoder para predecir la clase.
        """
        # Normalización
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Patching y Encoder
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Clasificador: aplanar y proyectar a clases
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # [batch, num_classes]
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Punto de entrada principal. Ejecuta la tarea correspondiente.

        x_enc: Serie de entrada [batch, seq_len, channels]
        x_mark_enc: Marcas temporales de entrada
        x_dec, x_mark_dec: Entrada/marcas del decodificador (no usadas aquí)
        mask: Máscara para imputación (1=observado, 0=faltante)
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, seq_len, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, seq_len, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, num_classes]
        return None
