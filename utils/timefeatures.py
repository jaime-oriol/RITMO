"""
Features temporales para encoding de timestamps.
Convierte fechas en características numéricas [-0.5, 0.5].
Adaptado de GluonTS (Amazon) para compatibilidad con modelos.

Cada feature captura periodicidad diferente:
- SecondOfMinute, MinuteOfHour, HourOfDay (intra-día)
- DayOfWeek, DayOfMonth, DayOfYear (intra-año)
- MonthOfYear, WeekOfYear (estacionalidad)
"""

# From: gluonts/src/gluonts/time_feature/_base.py
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    """Clase base para features temporales."""
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Extrae feature del índice temporal."""
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Segundo del minuto normalizado a [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minuto de la hora normalizado a [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hora del día normalizada a [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Día de la semana normalizado a [-0.5, 0.5]. Lunes=0, Domingo=6."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Día del mes normalizado a [-0.5, 0.5]. Día 1=inicio."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Día del año normalizado a [-0.5, 0.5]. Día 1=1 enero."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Mes del año normalizado a [-0.5, 0.5]. Enero=0."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Semana del año normalizada a [-0.5, 0.5]. Semana 1=primera."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Retorna features temporales apropiados para frecuencia dada.

    Mapea frecuencia de datos a conjunto de features relevantes.
    Frecuencias más finas necesitan más features.

    Args:
        freq_str: String de frecuencia tipo "[múltiplo][granularidad]"
                  Ejemplos: "12H", "5min", "1D", "W", "M"

    Returns:
        Lista de instancias TimeFeature

    Raises:
        RuntimeError: Si frecuencia no soportada

    Ejemplos:
        - "H" (horario) → [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
        - "D" (diario) → [DayOfWeek, DayOfMonth, DayOfYear]
        - "W" (semanal) → [DayOfMonth, WeekOfYear]
    """
    # Mapeo frecuencia → features relevantes
    features_by_offsets = {
        offsets.YearEnd: [],  # Anual: sin features (muy poca variación)
        offsets.QuarterEnd: [MonthOfYear],  # Trimestral
        offsets.MonthEnd: [MonthOfYear],  # Mensual
        offsets.Week: [DayOfMonth, WeekOfYear],  # Semanal
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],  # Diario
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],  # Días hábiles
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],  # Horario
        offsets.Minute: [  # Por minuto
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [  # Por segundo
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)  # Parsear string de frecuencia

    # Buscar tipo de offset y retornar features correspondientes
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    # Error si frecuencia no reconocida
    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    """
    Extrae todas las features temporales para conjunto de fechas.

    Args:
        dates: pd.DatetimeIndex con timestamps
        freq: String de frecuencia (default: 'h' horario)

    Returns:
        np.ndarray [num_features, len(dates)] con valores en [-0.5, 0.5]
    """
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
