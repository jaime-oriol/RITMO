"""
Clase base de experimentación.
Define interfaz común para todos los experimentos y gestiona el dispositivo (CPU/GPU).
"""

import os  # Operaciones del sistema
import torch  # Framework de deep learning
from models import DLinear, PatchTST, TimeMixer, TimeXer, TransformerCommon  # Modelos disponibles


class Exp_Basic(object):
    """
    Clase base para experimentos de series temporales.
    Gestiona: selección de dispositivo, registro de modelos, y define interfaz.
    """
    def __init__(self, args):
        """
        Inicializa experimento.

        args: Namespace con configuración (model, use_gpu, etc.)
        """
        self.args = args

        # Diccionario de modelos disponibles
        self.model_dict = {
            'DLinear': DLinear,    # Descomposición + Linear
            'PatchTST': PatchTST,  # Transformer con patches
            'TimeMixer': TimeMixer,  # Multi-scale mixing
            'TimeXer': TimeXer,    # Variables exógenas
            'TransformerCommon': TransformerCommon,  # Transformer común Plan A
        }

        # Configurar dispositivo y construir modelo
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        """Construye modelo. Debe implementarse en subclases."""
        raise NotImplementedError
        return None

    def _acquire_device(self):
        """
        Selecciona dispositivo de cómputo.
        Soporta: CUDA (GPU NVIDIA), MPS (GPU Apple), CPU
        """
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            # GPU NVIDIA
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            # GPU Apple Silicon
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            # CPU
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        """Carga datos. Implementar en subclases."""
        pass

    def vali(self):
        """Validación. Implementar en subclases."""
        pass

    def train(self):
        """Entrenamiento. Implementar en subclases."""
        pass

    def test(self):
        """Testing. Implementar en subclases."""
        pass
