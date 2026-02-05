"""
Experimentos Plan A: Comparación Controlada de Técnicas de Tokenización.

Compara 6 técnicas de tokenización (Discretización, Text-based, Patching,
Descomposición, Foundation, HMM) usando el MISMO Transformer para todas.

Objetivo: Aislar el efecto de la técnica de tokenización manteniendo todo
lo demás constante (RevIN, Transformer, protocolo de entrenamiento).

Pipeline por técnica:
    Data cruda → RevIN normalize → Tokenize → Embed → TransformerCommon
              → Predicción norm → RevIN denormalize → Predicción final

Referencias:
    - Kim et al. 2022 (RevIN normalization)
    - Anteproyecto RITMO (metodología Plan A)
"""

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider  # Factory para datasets
from exp.exp_basic import Exp_Basic  # Clase base de experimentos
from models.TransformerCommon import Model as TransformerCommon  # Transformer único
from layers.StandardNorm import Normalize  # RevIN oficial
from utils.tools import EarlyStopping, adjust_learning_rate, visual  # Utilidades
from utils.metrics import metric  # MSE, MAE

# Importar técnicas de tokenización
from tecnicas.discretization import sax_discretize, decode_sax
from tecnicas.text_based import text_based_tokenize, decode_text_based
from tecnicas.patching import patching_tokenize
from tecnicas.decomposition import decomposition_tokenize
from tecnicas.foundation import foundation_tokenize

# Importar embeddings naturales por técnica
from embeddings.embedding_generator import EmbeddingGenerator  # HMM
from embeddings.technique_embeddings import (
    DiscretizationEmbedding,  # SAX
    TextBasedEmbedding,       # LLMTime
    PatchingEmbedding,        # PatchTST
    DecompositionEmbedding,   # DLinear/Autoformer
    FoundationEmbedding       # MOMENT
)

# Importar HMM para tokenización probabilística
from hmm import baum_welch, viterbi_decode, load_hmm_params

warnings.filterwarnings('ignore')


class Exp_Plan_A(Exp_Basic):
    """
    Experimento Plan A: Comparación controlada de 6 técnicas de tokenización.

    Hereda de Exp_Basic para reutilizar infraestructura común pero implementa
    pipeline específico con:
    - RevIN externo (no en el modelo)
    - Tokenización por técnica
    - Embeddings naturales por técnica
    - Transformer único para todas

    Técnicas soportadas:
        1. 'discretization' - SAX (Lin et al. 2007)
        2. 'text_based' - LLMTime (Gruver et al. 2023)
        3. 'patching' - PatchTST (Nie et al. 2023)
        4. 'decomposition' - DLinear/Autoformer (Zeng et al. 2023, Wu et al. 2021)
        5. 'foundation' - MOMENT (Goswami et al. 2024)
        6. 'hmm' - RITMO (propuesta TFG)
    """

    def __init__(self, args):
        """
        Inicializa experimento Plan A.

        Args:
            args: Configuración con atributos:
                - technique: Técnica de tokenización a usar
                - model: Debe ser 'TransformerCommon'
                - data: Dataset (ETTh1, ETTh2, Weather, Electricity, Traffic, Exchange)
                - seq_len: Longitud de entrada (default: 96)
                - pred_len: Horizonte de predicción (96, 192, 336, 720)
                - d_model: Dimensión embeddings (default: 128)
                - n_heads: Attention heads (default: 4)
                - e_layers: Capas encoder (default: 2)
                - otros parámetros de entrenamiento
        """
        super(Exp_Plan_A, self).__init__(args)

        # Validar técnica especificada
        valid_techniques = ['discretization', 'text_based', 'patching',
                           'decomposition', 'foundation', 'hmm']
        if args.technique not in valid_techniques:
            raise ValueError(f"Técnica '{args.technique}' no válida. "
                           f"Opciones: {valid_techniques}")

        # Guardar técnica seleccionada
        self.technique = args.technique

        # Inicializar RevIN (Normalize de layers/StandardNorm.py)
        # num_features=1 para forecasting univariado
        # Se usa para normalizar ANTES de tokenizar
        self.revin = Normalize(
            num_features=1,  # Univariado (args.enc_in debería ser 1)
            eps=1e-5,
            affine=False,  # Sin parámetros aprendibles (normalización simple)
            subtract_last=False,  # Usar media, no último valor
            non_norm=False  # Sí normalizar (no bypass)
        )

        # Inicializar embedder según técnica
        # Cada técnica tiene su embedding natural
        self._init_embedder()

        # Cargar parámetros HMM si la técnica es 'hmm'
        if self.technique == 'hmm':
            self._load_hmm_params()

    def _init_embedder(self):
        """
        Inicializa el embedder según la técnica seleccionada.

        Cada técnica tiene su embedding natural:
        - Discretización: Lookup table aprendible
        - Text-based: Character embeddings
        - Patching: Proyección lineal + positional
        - Descomposición: Proyección por componente
        - Foundation: Patch + mask token
        - HMM: Embeddings estructurados [μ, σ, A]
        """
        d_model = self.args.d_model

        if self.technique == 'discretization':
            # SAX: 8 símbolos → embedding table [8, d_model]
            self.embedder = DiscretizationEmbedding(
                vocab_size=8,  # Alfabeto SAX estándar
                d_model=d_model
            )

        elif self.technique == 'text_based':
            # LLMTime: caracteres → embeddings [14, d_model]
            self.embedder = TextBasedEmbedding(d_model=d_model)

        elif self.technique == 'patching':
            # PatchTST: patches [P=16] → embeddings [d_model]
            self.embedder = PatchingEmbedding(
                patch_len=16,  # Longitud de patch estándar
                d_model=d_model
            )

        elif self.technique == 'decomposition':
            # DLinear/Autoformer: trend + seasonal → embeddings
            self.embedder = DecompositionEmbedding(d_model=d_model)

        elif self.technique == 'foundation':
            # MOMENT: patches + masking → embeddings
            self.embedder = FoundationEmbedding(
                patch_len=16,
                d_model=d_model
            )

        elif self.technique == 'hmm':
            # RITMO: estados k → embeddings estructurados [μ_k, σ_k, A[k,:]]
            # Se inicializa después de cargar parámetros HMM
            pass  # Se crea en _load_hmm_params()

        # Mover embedder a dispositivo (GPU si está disponible)
        if hasattr(self, 'embedder'):
            self.embedder = self.embedder.to(self.device)

    def _load_hmm_params(self):
        """
        Carga parámetros HMM pre-entrenados desde cache.

        Los HMM se entrenan previamente con Baum-Welch sobre:
        - ETTh1, ETTh2, Weather, Electricity (4 datasets train)
        - K=5 estados ocultos

        Cache ubicado en: cache/hmm_{dataset}_K5.pth
        """
        dataset_name = self.args.data.lower()  # ej: 'etth1', 'etth2'
        K = 5  # Número de estados (fijo según metodología)

        # Ruta al archivo de cache
        cache_path = f'./cache/hmm_{dataset_name}_K{K}.pth'

        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No se encontró cache HMM en {cache_path}. "
                f"Entrena el HMM primero con hmm/baum_welch.py"
            )

        # Cargar parámetros (A, mu, sigma, pi)
        self.hmm_params = load_hmm_params(cache_path)

        print(f'[HMM] Parámetros cargados desde {cache_path}')
        print(f'  Estados: K={K}')
        print(f'  mu (medias): {self.hmm_params["mu"]}')
        print(f'  sigma (stds): {self.hmm_params["sigma"]}')

        # Inicializar embedder HMM con parámetros cargados
        self.embedder = EmbeddingGenerator(
            hmm_params=self.hmm_params,
            d_model=self.args.d_model,
            device=self.device
        )

    def _build_model(self):
        """
        Construye el modelo TransformerCommon.

        Este Transformer es ÚNICO para todas las técnicas (comparación justa).
        Solo recibe embeddings, no hace RevIN interno.

        Returns:
            TransformerCommon con arquitectura fija
        """
        model = TransformerCommon(self.args).float()

        # Multi-GPU si está disponible
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _tokenize_batch(self, batch_norm):
        """
        Tokeniza batch de datos normalizados según técnica seleccionada.

        Args:
            batch_norm: Datos normalizados [batch, seq_len, 1]

        Returns:
            Tokens según la técnica (formato varía por técnica)
        """
        # Convertir de tensor PyTorch a numpy para funciones de tokenización
        # [batch, seq_len, 1] → [batch, seq_len]
        batch_np = batch_norm.squeeze(-1).cpu().numpy()

        batch_tokens = []

        # Tokenizar cada muestra del batch
        for i in range(batch_np.shape[0]):
            serie = batch_np[i]  # [seq_len]

            if self.technique == 'discretization':
                # SAX: valores continuos → símbolos discretos
                result = sax_discretize(serie, alphabet_size=8)
                tokens = result['tokens']  # [seq_len] con valores 0-7

            elif self.technique == 'text_based':
                # LLMTime: valores → strings separados por espacios
                result = text_based_tokenize(serie, precision=2)
                tokens = result['tokens_per_value']  # Lista de strings

            elif self.technique == 'patching':
                # PatchTST: serie → patches no solapados
                patches = patching_tokenize(serie, patch_len=16, stride=16)
                tokens = patches  # [num_patches, 16]

            elif self.technique == 'decomposition':
                # DLinear/Autoformer: serie → trend + seasonal
                result = decomposition_tokenize(serie, kernel_size=25)
                tokens = result  # Dict con 'trend' y 'seasonal'

            elif self.technique == 'foundation':
                # MOMENT: patches + masking
                result = foundation_tokenize(serie, patch_len=16, stride=16, mask_ratio=0.0)
                tokens = result  # Dict con 'patches', 'mask', etc.

            elif self.technique == 'hmm':
                # RITMO: Viterbi decode → estados óptimos
                states, _ = viterbi_decode(
                    serie,
                    self.hmm_params['A'],
                    self.hmm_params['pi'],
                    self.hmm_params['mu'],
                    self.hmm_params['sigma']
                )
                tokens = states  # [seq_len] con valores 0-(K-1)

            batch_tokens.append(tokens)

        return batch_tokens

    def _embed_tokens(self, batch_tokens):
        """
        Convierte tokens en embeddings usando embedder natural de la técnica.

        Args:
            batch_tokens: Lista de tokens (formato varía por técnica)

        Returns:
            Embeddings [batch, seq_len_tokens, d_model]
            NOTA: seq_len_tokens varía por técnica (ej: patching tiene menos)
        """
        batch_embeds = []

        for tokens in batch_tokens:
            if self.technique == 'discretization':
                # tokens: [seq_len] con valores 0-7
                # embedding: [seq_len, d_model]
                tokens_tensor = torch.from_numpy(tokens).long().to(self.device)
                embeds = self.embedder(tokens_tensor)

            elif self.technique == 'text_based':
                # tokens: lista de strings
                # embedding: [seq_len, d_model] (promedia caracteres por timestep)
                embeds = self.embedder(tokens)  # TextBasedEmbedding.forward()

            elif self.technique == 'patching':
                # tokens: [num_patches, patch_len]
                # embedding: [num_patches, d_model]
                patches_tensor = torch.from_numpy(tokens).float().to(self.device)
                embeds = self.embedder(patches_tensor)

            elif self.technique == 'decomposition':
                # tokens: dict con 'trend' y 'seasonal'
                # embedding: [2, d_model] o [seq_len, d_model] según implementación
                trend = torch.from_numpy(tokens['trend']).float().to(self.device)
                seasonal = torch.from_numpy(tokens['seasonal']).float().to(self.device)
                embeds = self.embedder(trend, seasonal)

            elif self.technique == 'foundation':
                # tokens: dict con 'patches'
                # embedding: [num_patches, d_model]
                patches_tensor = torch.from_numpy(tokens['patches']).float().to(self.device)
                mask_tensor = torch.from_numpy(tokens['mask']).bool().to(self.device)
                embeds = self.embedder(patches_tensor, mask_tensor)

            elif self.technique == 'hmm':
                # tokens: [seq_len] estados 0-(K-1)
                # embedding: [seq_len, d_model] estructurado [μ, σ, A]
                embeds = self.embedder.forward(tokens)

            batch_embeds.append(embeds)

        # Stack en batch dimension
        # NOTA: Adaptive pooling en TransformerCommon maneja diferentes seq_len
        batch_embeds_tensor = torch.stack(batch_embeds, dim=0)

        return batch_embeds_tensor

    def _get_data(self, flag):
        """
        Obtiene DataLoader para train/val/test.

        Reutiliza data_factory de Time-Series-Library.

        Args:
            flag: 'train', 'val', o 'test'

        Returns:
            data_set, data_loader
        """
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """Selecciona optimizador (Adam por defecto)."""
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """Selecciona función de pérdida (MSE por defecto)."""
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """
        Validación en un epoch.

        Pipeline:
            1. RevIN normalize batch
            2. Tokenizar
            3. Embed
            4. Forward TransformerCommon
            5. RevIN denormalize salida
            6. Calcular loss

        Args:
            vali_data: Dataset de validación
            vali_loader: DataLoader de validación
            criterion: Función de pérdida

        Returns:
            total_loss: Loss promedio en validación
        """
        total_loss = []
        self.model.eval()  # Modo evaluación (desactiva dropout)

        with torch.no_grad():  # No calcular gradientes (más rápido)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # Mover datos a dispositivo
                batch_x = batch_x.float().to(self.device)  # [batch, seq_len, enc_in]
                batch_y = batch_y.float().to(self.device)  # [batch, pred_len, enc_in]

                # === PASO 1: RevIN NORMALIZE ===
                # Normalizar datos crudos ANTES de tokenizar
                batch_x_norm = self.revin(batch_x, mode='norm')

                # === PASO 2: TOKENIZAR ===
                # Convertir datos normalizados a tokens según técnica
                batch_tokens = self._tokenize_batch(batch_x_norm)

                # === PASO 3: EMBED ===
                # Convertir tokens a embeddings [batch, seq_len_tokens, d_model]
                batch_embeds = self._embed_tokens(batch_tokens)

                # === PASO 4: TRANSFORMER ===
                # Forward pass - salida en espacio normalizado
                outputs_norm = self.model(batch_embeds)  # [batch, pred_len, enc_in]

                # === PASO 5: RevIN DENORMALIZE ===
                # Desnormalizar predicción a espacio original
                outputs = self.revin(outputs_norm, mode='denorm')

                # Ground truth (ya está en espacio original)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Calcular loss
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()  # Volver a modo entrenamiento
        return total_loss

    def train(self, setting):
        """
        Entrena el modelo.

        Pipeline por epoch:
            1. Cargar batch
            2. RevIN normalize
            3. Tokenizar según técnica
            4. Embed con embedder natural
            5. Forward TransformerCommon
            6. RevIN denormalize salida
            7. Calcular loss y backprop
            8. Validar

        Args:
            setting: String identificador del experimento
        """
        # Obtener dataloaders
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Directorio para guardar checkpoints
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # Utilidades de entrenamiento
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # Training loop
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()  # Modo entrenamiento
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()  # Reset gradientes

                # Mover datos a dispositivo
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # === PIPELINE PLAN A ===

                # 1. RevIN normalize (sobre datos crudos)
                batch_x_norm = self.revin(batch_x, mode='norm')

                # 2. Tokenizar (según técnica)
                batch_tokens = self._tokenize_batch(batch_x_norm)

                # 3. Embed (con embedder natural)
                batch_embeds = self._embed_tokens(batch_tokens)

                # 4. Forward Transformer
                outputs_norm = self.model(batch_embeds)

                # 5. RevIN denormalize
                outputs = self.revin(outputs_norm, mode='denorm')

                # Ground truth
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Loss y backprop
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                # Logging cada 100 iteraciones
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                # Backpropagation
                loss.backward()
                model_optim.step()

            print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch+1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} "
                  f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            # Early stopping
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Ajustar learning rate
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # Cargar mejor modelo
        best_model_path = path + '/checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """
        Evalúa el modelo en test set.

        Calcula MSE y MAE, guarda predicciones y ground truth.

        Args:
            setting: Identificador del experimento
            test: Si 0 usa test loader, si 1 usa val loader

        Returns:
            None (imprime métricas y guarda resultados)
        """
        test_data, test_loader = self._get_data(flag='test' if test == 0 else 'val')

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # Pipeline completo
                batch_x_norm = self.revin(batch_x, mode='norm')
                batch_tokens = self._tokenize_batch(batch_x_norm)
                batch_embeds = self._embed_tokens(batch_tokens)
                outputs_norm = self.model(batch_embeds)
                outputs = self.revin(outputs_norm, mode='denorm')

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                # Visualizar algunos ejemplos
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # Resultado
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'Technique: {self.technique} | MSE: {mse:.7f}, MAE: {mae:.7f}')

        # Guardar métricas
        f = open(os.path.join(folder_path, 'result.txt'), 'a')
        f.write(setting + "  \n")
        f.write(f'Technique: {self.technique} | MSE: {mse:.7f}, MAE: {mae:.7f}')
        f.write('\n')
        f.write('\n')
        f.close()

        # Guardar predicciones
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
