"""
DLinear - Modelo baseline simple pero efectivo.
Idea: Descomponer la serie en tendencia + estacionalidad, y predecir cada una con una capa lineal.
Paper: https://arxiv.org/pdf/2205.13504.pdf (Zeng et al., 2023)
"""

import torch  # Framework de deep learning
import torch.nn as nn  # Módulo de redes neuronales
from layers.Autoformer_EncDec import series_decomp  # Descomposición tendencia/estacionalidad


class Model(nn.Module):
    """
    DLinear: Descomposición + Linear.
    Separa la serie en tendencia y estacionalidad, predice cada una por separado,
    y suma los resultados. Sorprendentemente simple pero supera a muchos Transformers.
    """

    def __init__(self, configs, individual=False):
        """
        Inicializa el modelo.

        configs: Configuración con seq_len, pred_len, enc_in, etc.
        individual: Si True, cada variable tiene su propia capa lineal
                   Si False, todas comparten la misma capa
        """
        super(Model, self).__init__()

        # Guardar configuración
        self.task_name = configs.task_name  # Tipo de tarea (forecast, classification, etc.)
        self.seq_len = configs.seq_len  # Longitud de entrada

        # Longitud de predicción según la tarea
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len  # Misma longitud que entrada
        else:
            self.pred_len = configs.pred_len  # Horizonte de predicción

        # Bloque de descomposición: separa tendencia de estacionalidad
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual  # ¿Capas individuales por variable?
        self.channels = configs.enc_in  # Número de variables/canales

        # Crear capas lineales para predecir estacionalidad y tendencia
        if self.individual:
            # Modo individual: cada variable tiene sus propias capas
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                # Capa lineal: entrada seq_len → salida pred_len
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                # Inicializar pesos como promedio (1/seq_len)
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            # Modo compartido: todas las variables usan las mismas capas
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Inicializar pesos como promedio
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # Capa extra para clasificación
        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        """
        Procesa la entrada: descompone y predice cada componente.

        x: Tensor [batch, seq_len, channels]
        Salida: Tensor [batch, pred_len, channels]
        """
        # Paso 1: Descomponer en estacionalidad y tendencia
        seasonal_init, trend_init = self.decompsition(x)

        # Cambiar dimensiones para las capas lineales: [B, C, T]
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        if self.individual:
            # Procesar cada canal por separado
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            # Procesar todos los canales con las mismas capas
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # Paso 2: Sumar componentes para obtener predicción final
        x = seasonal_output + trend_output

        # Volver a dimensiones originales: [B, T, C]
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        """Predicción de series temporales."""
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        """Rellenar valores faltantes."""
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        """Detectar anomalías."""
        return self.encoder(x_enc)

    def classification(self, x_enc):
        """Clasificar la serie temporal."""
        enc_out = self.encoder(x_enc)
        # Aplanar: [batch, seq_len, channels] → [batch, seq_len * channels]
        output = enc_out.reshape(enc_out.shape[0], -1)
        # Proyectar a clases
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Punto de entrada principal del modelo.

        x_enc: Entrada codificada [batch, seq_len, channels]
        x_mark_enc: Marcas temporales (no usadas en DLinear)
        x_dec: Entrada decodificador (no usada en DLinear)
        x_mark_dec: Marcas temporales decodificador (no usadas)
        mask: Máscara para imputación
        """
        # Ejecutar según el tipo de tarea
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
