"""
Métricas de evaluación para series temporales.
Incluye: MSE, MAE, RMSE, MAPE, MSPE, RSE, CORR.
"""

import numpy as np  # Operaciones numéricas


def RSE(pred, true):
    """
    Relative Squared Error.
    Normaliza el error por la varianza de los datos reales.
    """
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """
    Correlación de Pearson entre predicción y valores reales.
    Mide la relación lineal entre ambas series.
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    d = np.where(d == 0, 1.0, d)  # Evitar división por cero
    return (u / d).mean(-1)


def MAE(pred, true):
    """
    Mean Absolute Error.
    Promedio de errores absolutos: más robusto a outliers que MSE.
    """
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    """
    Mean Squared Error.
    Métrica principal del TFG: promedio de errores al cuadrado.
    """
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    """
    Root Mean Squared Error.
    Raíz de MSE: mismas unidades que los datos.
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """
    Mean Absolute Percentage Error.
    Error porcentual: útil para comparar entre datasets.
    """
    eps = 1e-8
    return np.mean(np.abs((true - pred) / np.maximum(np.abs(true), eps)))


def MSPE(pred, true):
    """
    Mean Squared Percentage Error.
    Versión cuadrada de MAPE.
    """
    eps = 1e-8
    return np.mean(np.square((true - pred) / np.maximum(np.abs(true), eps)))


def metric(pred, true):
    """
    Calcula todas las métricas principales.

    Args:
        pred: Predicciones [N, T, C]
        true: Valores reales [N, T, C]

    Returns:
        Tupla (mae, mse, rmse, mape, mspe)
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
