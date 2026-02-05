"""
Herramientas de utilidad para entrenamiento y visualización.
Incluye: EarlyStopping, ajuste de learning rate, visualización.
"""

import os  # Sistema de archivos
import numpy as np  # Operaciones numéricas
import torch  # Deep learning
import matplotlib.pyplot as plt  # Visualización
import pandas as pd  # DataFrames
import math  # Funciones matemáticas

plt.switch_backend('agg')  # Backend sin GUI para servidores


def adjust_learning_rate(optimizer, epoch, args):
    """
    Ajusta learning rate según esquema especificado.

    Esquemas:
    - type1: Decay exponencial (0.5 cada epoch)
    - type2: Manual por epoch (predefinido)
    - type3: Decay suave después de epoch 3
    - cosine: Cosine annealing
    """
    if args.lradj == 'type1':
        # Decay exponencial agresivo: cada epoch multiplica por 0.5
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        # Schedule manual: LR predefinido para cada milestone
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        # Warmup 3 epochs, luego decay geométrico 0.9 por epoch
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        # Cosine annealing: curva suave de 1.0 a 0.0
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}

    # Aplicar nuevo LR si corresponde este epoch
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]  # Obtener LR para este epoch
        for param_group in optimizer.param_groups:  # Iterar grupos de parámetros
            param_group['lr'] = lr  # Actualizar LR de cada grupo
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    """
    Early stopping para evitar overfitting.
    Detiene entrenamiento si no mejora en `patience` epochs.
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        patience: Epochs sin mejora antes de parar
        verbose: Si True, imprime cuando guarda
        delta: Mejora mínima requerida
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0  # Epochs sin mejora
        self.best_score = None
        self.early_stop = False  # Flag de parada
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """Evalúa si debe parar y guarda mejor modelo."""
        score = -val_loss  # Negativo porque menor loss es mejor (score más alto = mejor)
        if self.best_score is None:
            # Primera época: inicializar y guardar
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            # No mejoró (score actual < mejor + delta mínimo)
            self.counter += 1  # Incrementar contador de epochs sin mejora
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                # Alcanzado el límite: activar flag de parada
                self.early_stop = True
        else:
            # Mejoró suficientemente: guardar y resetear
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0  # Resetear contador

    def save_checkpoint(self, val_loss, model, path):
        """Guarda modelo cuando val_loss mejora."""
        if self.verbose:
            # Informar mejora si modo verbose activado
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # Guardar state_dict (pesos del modelo) en disco
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        # Actualizar mejor loss registrado
        self.val_loss_min = val_loss


class dotdict(dict):
    """
    Diccionario con acceso por punto.
    Permite: args.param en vez de args['param']
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    """
    Normalizador z-score simple.
    No confundir con sklearn.preprocessing.StandardScaler.
    """
    def __init__(self, mean, std):
        """
        mean: Media para centrar
        std: Desviación estándar para escalar
        """
        self.mean = mean
        self.std = std

    def transform(self, data):
        """Normaliza: (x - mean) / std"""
        return (data - self.mean) / self.std  # z-score normalización

    def inverse_transform(self, data):
        """Desnormaliza: x * std + mean"""
        return (data * self.std) + self.mean  # Revertir normalización


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Visualiza predicción vs ground truth.

    true: Serie real
    preds: Serie predicha (opcional)
    name: Ruta de guardado
    """
    plt.figure()  # Crear nueva figura
    if preds is not None:
        # Graficar predicción si se proporciona
        plt.plot(preds, label='Prediction', linewidth=2)
    # Graficar serie real (siempre)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()  # Añadir leyenda
    plt.savefig(name, bbox_inches='tight')  # Guardar sin whitespace extra


def adjustment(gt, pred):
    """
    Ajuste para detección de anomalías.
    Si se detecta una anomalía correctamente, extiende la detección
    a todo el segmento anómalo continuo.
    """
    anomaly_state = False  # Flag para trackear si estamos en segmento anómalo
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            # Detectado inicio de anomalía: primera detección correcta
            anomaly_state = True
            # Propagar hacia atrás: marcar todo el segmento anómalo previo
            for j in range(i, 0, -1):
                if gt[j] == 0:  # Fin del segmento anómalo
                    break
                else:
                    if pred[j] == 0:  # Marcar como detectado
                        pred[j] = 1
            # Propagar hacia adelante: marcar resto del segmento anómalo
            for j in range(i, len(gt)):
                if gt[j] == 0:  # Fin del segmento anómalo
                    break
                else:
                    if pred[j] == 0:  # Marcar como detectado
                        pred[j] = 1
        elif gt[i] == 0:
            # Salimos del segmento anómalo: resetear flag
            anomaly_state = False
        if anomaly_state:
            # Dentro de segmento anómalo: asegurar marcado
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    """Accuracy: proporción de predicciones correctas."""
    return np.mean(y_pred == y_true)  # Promedio de aciertos (True=1, False=0)
