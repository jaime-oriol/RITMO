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
        # Decay exponencial agresivo
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        # Schedule manual
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        # Warmup 3 epochs, luego decay suave
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        # Cosine annealing
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}

    # Aplicar nuevo LR
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
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
        score = -val_loss  # Negativo porque menor loss es mejor
        if self.best_score is None:
            # Primera vez
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            # No mejoró lo suficiente
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Mejoró: guardar y resetear contador
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """Guarda modelo cuando val_loss mejora."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
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
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """Desnormaliza: x * std + mean"""
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Visualiza predicción vs ground truth.

    true: Serie real
    preds: Serie predicha (opcional)
    name: Ruta de guardado
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    """
    Ajuste para detección de anomalías.
    Si se detecta una anomalía correctamente, extiende la detección
    a todo el segmento anómalo continuo.
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            # Detectado inicio de anomalía
            anomaly_state = True
            # Propagar hacia atrás
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            # Propagar hacia adelante
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    """Accuracy: proporción de predicciones correctas."""
    return np.mean(y_pred == y_true)
