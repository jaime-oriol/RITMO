"""
Experimento Plan A: Comparación controlada de 6 técnicas de tokenización.

Implementa pipeline completo:
    1. RevIN normalización (externa, compartida)
    2. Tokenización (variable según técnica)
    3. Embeddings naturales (específicos por técnica)
    4. TransformerCommon (arquitectura FIJA)
    5. RevIN desnormalización

Técnicas soportadas:
    - discretization: SAX, VQ-VAE (símbolos discretos)
    - text_based: LLMTime (serialización a caracteres)
    - patching: PatchTST (segmentación en patches)
    - decomposition: DLinear/Autoformer (trend + seasonal)
    - foundation: MOMENT (masked patches)
    - hmm: RITMO hard (Viterbi argmax → embedding lookup)
    - hmm_soft: RITMO soft (gamma posteriors → mezcla ponderada de embeddings)
"""

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from layers.StandardNorm import Normalize  # RevIN oficial
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

# Importar técnicas de tokenización
from tecnicas.discretization import sax_discretize
from tecnicas.text_based import text_based_tokenize
from tecnicas.patching import patching_tokenize
from tecnicas.decomposition import decomposition_tokenize
from tecnicas.foundation import foundation_tokenize

# Importar HMM para técnica RITMO
from hmm import viterbi_decode, forward_backward
from embeddings import EmbeddingGenerator

warnings.filterwarnings('ignore')


class Exp_Plan_A(Exp_Basic):
    """
    Experimento Plan A: Comparación justa de técnicas de tokenización.

    Garantiza comparación controlada manteniendo TODO fijo excepto tokenización:
        - RevIN: Externo, idéntico para todas las técnicas
        - Transformer: Arquitectura fija (TransformerCommon)
        - Hiperparámetros: d_model=128, n_heads=4, e_layers=2, d_ff=256
        - Train/val/test: Mismos splits para todas las técnicas
    """

    def __init__(self, args):
        """
        Inicializa experimento Plan A.

        Args:
            args: Argumentos con técnica de tokenización en args.technique
        """
        super(Exp_Plan_A, self).__init__(args)

        # Verificar técnica válida
        valid_techniques = ['discretization', 'text_based', 'patching', 'decomposition', 'foundation', 'hmm', 'hmm_soft']
        if not hasattr(args, 'technique') or args.technique not in valid_techniques:
            raise ValueError(f"args.technique debe ser una de: {valid_techniques}")

        self.technique = args.technique

        # Inicializar RevIN (Normalize de layers/StandardNorm.py)
        # CRÍTICO: Externa, compartida por todas las técnicas para comparación justa
        self.revin = Normalize(
            num_features=args.enc_in,  # Número de features (1 para univariado)
            eps=1e-5,
            affine=False,  # Sin parámetros aprendibles (no queremos que aprenda escala)
            subtract_last=False,  # Usar media, no último valor
            non_norm=False  # Sí normalizar
        )

        # Inicializar embedder según técnica
        self._init_embedder()

    def _init_embedder(self):
        """
        Inicializa embedder natural según técnica de tokenización.

        Cada técnica tiene su propia forma natural de generar embeddings:
            - discretization: Lookup table para símbolos discretos
            - text_based: Character embeddings
            - patching: Proyección lineal de patches
            - decomposition: Proyección por componente
            - foundation: Patch + position + mask embeddings
            - hmm: Embeddings estructurados [μ_k, σ_k, A[k,:]]
        """
        d_model = self.args.d_model

        if self.technique == 'discretization':
            # Lookup table para 8 símbolos SAX
            vocab_size = getattr(self.args, 'sax_vocab_size', 8)
            self.embedder = nn.Embedding(vocab_size, d_model)

        elif self.technique == 'text_based':
            # Character embeddings (0-9, signo, punto, espacio)
            vocab_size = getattr(self.args, 'char_vocab_size', 13)
            self.embedder = nn.Embedding(vocab_size, d_model)

        elif self.technique == 'patching':
            # Proyección lineal desde patches
            patch_len = getattr(self.args, 'patch_len', 16)
            self.embedder = nn.Linear(patch_len, d_model)

        elif self.technique == 'decomposition':
            # Dos proyecciones: una para trend, otra para seasonal
            self.embedder_trend = nn.Linear(1, d_model // 2)
            self.embedder_seasonal = nn.Linear(1, d_model // 2)

        elif self.technique == 'foundation':
            # Patch + position + mask token embeddings
            patch_len = getattr(self.args, 'patch_len', 16)
            self.embedder_patch = nn.Linear(patch_len, d_model)
            self.embedder_pos = nn.Parameter(torch.zeros(1, 5000, d_model))  # Max 5000 patches
            self.embedder_mask = nn.Parameter(torch.zeros(1, 1, d_model))  # [MASK] token

        elif self.technique in ('hmm', 'hmm_soft'):
            # EmbeddingGenerator con parámetros HMM
            # Cache debe estar en ./cache/hmm_{data}_{K}.pth
            cache_path = f'./cache/hmm_{self.args.data.lower()}_K5.pth'
            if not os.path.exists(cache_path):
                raise FileNotFoundError(f"Cache HMM no encontrado: {cache_path}. Ejecutar primero entrenamiento HMM.")

            hmm_params = torch.load(cache_path, weights_only=True)
            self.embedder = EmbeddingGenerator(
                hmm_params=hmm_params,
                d_model=d_model,
                device=self.device
            )
            # Guardar parámetros HMM para Viterbi / forward-backward
            self.hmm_params = hmm_params

    def _build_model(self):
        """
        Construye TransformerCommon (arquitectura fija para Plan A).

        Returns:
            Modelo TransformerCommon con hiperparámetros fijos
        """
        # TransformerCommon está registrado en exp_basic.py
        model = self.model_dict['TransformerCommon'].Model(self.args).float()

        # Multi-GPU si está habilitado
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        """Carga dataset y dataloader. flag: 'train', 'val', o 'test'"""
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        Crea optimizador Adam.

        Optimiza parámetros del Transformer + embedder (NO RevIN, que no es aprendible).
        """
        # Parámetros del modelo + embedder
        params = list(self.model.parameters())

        # Añadir parámetros del embedder según técnica
        if self.technique in ['discretization', 'text_based', 'patching']:
            params += list(self.embedder.parameters())
        elif self.technique == 'decomposition':
            params += list(self.embedder_trend.parameters())
            params += list(self.embedder_seasonal.parameters())
        elif self.technique == 'foundation':
            params += list(self.embedder_patch.parameters())
            params += [self.embedder_pos, self.embedder_mask]
        elif self.technique in ('hmm', 'hmm_soft'):
            # EmbeddingGenerator tiene proyección lineal aprendible
            params += list(self.embedder.projection.parameters())

        model_optim = optim.Adam(params, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """Función de pérdida: MSE en espacio original (después de RevIN⁻¹)"""
        criterion = nn.MSELoss()
        return criterion

    def _tokenize_batch(self, batch_x_norm):
        """
        Tokeniza batch según técnica especificada.

        Args:
            batch_x_norm: Batch normalizado [B, L, C]

        Returns:
            Tokens según técnica (forma depende de técnica)
        """
        B, L, C = batch_x_norm.shape
        tokens_list = []

        for i in range(B):
            serie_norm = batch_x_norm[i, :, 0].detach().cpu().numpy()  # [L]

            if self.technique == 'discretization':
                # SAX: símbolos discretos
                result = sax_discretize(serie_norm, alphabet_size=8)
                tokens_list.append(torch.tensor(result['tokens'], dtype=torch.long))

            elif self.technique == 'text_based':
                # LLMTime: caracteres
                result = text_based_tokenize(serie_norm, precision=2)
                text = result['text']
                # Mapear caracteres a índices
                char_to_idx = {c: i for i, c in enumerate('0123456789-. ')}
                indices = [char_to_idx.get(c, 12) for c in text]  # 12 = unknown
                tokens_list.append(torch.tensor(indices, dtype=torch.long))

            elif self.technique == 'patching':
                # PatchTST: patches
                patches = patching_tokenize(serie_norm, patch_len=16, stride=16)
                tokens_list.append(torch.tensor(patches, dtype=torch.float32))

            elif self.technique == 'decomposition':
                # DLinear/Autoformer: trend + seasonal
                result = decomposition_tokenize(serie_norm, kernel_size=25)
                trend = result['trend']
                seasonal = result['seasonal']
                # Stack como [L, 2]
                components = np.stack([trend, seasonal], axis=-1)
                tokens_list.append(torch.tensor(components, dtype=torch.float32))

            elif self.technique == 'foundation':
                # MOMENT: patches con masking
                result = foundation_tokenize(serie_norm, patch_len=16, stride=16, mask_ratio=0.3)
                patches = result['patches']
                tokens_list.append(torch.tensor(patches, dtype=torch.float32))

            elif self.technique == 'hmm':
                # RITMO hard: Viterbi decoding (argmax)
                states, _ = viterbi_decode(
                    observations=serie_norm,
                    pi=self.hmm_params['pi'].cpu().numpy(),
                    A=self.hmm_params['A'].cpu().numpy(),
                    mu=self.hmm_params['mu'].cpu().numpy(),
                    sigma=self.hmm_params['sigma'].cpu().numpy()
                )
                tokens_list.append(torch.tensor(states, dtype=torch.long))

            elif self.technique == 'hmm_soft':
                # RITMO soft: gamma posteriors (mezcla ponderada)
                gamma, _, _ = forward_backward(
                    observations=serie_norm,
                    pi=self.hmm_params['pi'].cpu().numpy(),
                    A=self.hmm_params['A'].cpu().numpy(),
                    mu=self.hmm_params['mu'].cpu().numpy(),
                    sigma=self.hmm_params['sigma'].cpu().numpy()
                )
                tokens_list.append(torch.tensor(gamma, dtype=torch.float32))

        return tokens_list

    def _embed_tokens(self, tokens_list):
        """
        Genera embeddings desde tokens usando embedder natural.

        Args:
            tokens_list: Lista de tokens (uno por muestra en batch)

        Returns:
            Embeddings [B, seq_len, d_model]
        """
        embeddings_list = []

        for tokens in tokens_list:
            tokens = tokens.to(self.device)

            if self.technique in ['discretization', 'text_based']:
                # Lookup table: [seq_len] → [seq_len, d_model]
                embeds = self.embedder(tokens)

            elif self.technique == 'patching':
                # Proyección lineal: [num_patches, patch_len] → [num_patches, d_model]
                embeds = self.embedder(tokens)

            elif self.technique == 'decomposition':
                # Proyectar trend y seasonal por separado, concatenar
                trend = tokens[:, 0].unsqueeze(-1)  # [L, 1]
                seasonal = tokens[:, 1].unsqueeze(-1)  # [L, 1]
                embed_trend = self.embedder_trend(trend)  # [L, d_model//2]
                embed_seasonal = self.embedder_seasonal(seasonal)  # [L, d_model//2]
                embeds = torch.cat([embed_trend, embed_seasonal], dim=-1)  # [L, d_model]

            elif self.technique == 'foundation':
                # Patch + position embeddings
                patches = tokens  # [num_patches, patch_len]
                embeds = self.embedder_patch(patches)  # [num_patches, d_model]
                # Añadir positional
                embeds = embeds + self.embedder_pos[:, :embeds.shape[0], :]

            elif self.technique == 'hmm':
                # EmbeddingGenerator hard: [seq_len] → [seq_len, d_model]
                embeds = self.embedder(tokens)

            elif self.technique == 'hmm_soft':
                # EmbeddingGenerator soft: [seq_len, K] → [seq_len, d_model]
                embeds = self.embedder.forward_soft(tokens)

            embeddings_list.append(embeds)

        # Pad/truncate a longitud común si es necesario
        # Por ahora asumimos longitud fija o adaptive pooling maneja diferencias
        max_len = max(e.shape[0] for e in embeddings_list)
        padded = []
        for e in embeddings_list:
            if e.shape[0] < max_len:
                pad = torch.zeros(max_len - e.shape[0], e.shape[1], device=e.device)
                e = torch.cat([e, pad], dim=0)
            padded.append(e)

        # Stack: [B, seq_len, d_model]
        batch_embeds = torch.stack(padded, dim=0)
        return batch_embeds

    def vali(self, vali_data, vali_loader, criterion):
        """
        Evalúa modelo en conjunto de validación.

        Pipeline:
            1. RevIN normalize
            2. Tokenizar
            3. Embed
            4. Forward Transformer
            5. RevIN denormalize
            6. Calcular loss

        Returns:
            Pérdida promedio
        """
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # Mover a dispositivo
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # === PIPELINE PLAN A ===

                # 1. RevIN normalize (sobre datos crudos)
                batch_x_norm = self.revin(batch_x, mode='norm')

                # 2. Tokenizar (según técnica)
                batch_tokens = self._tokenize_batch(batch_x_norm)

                # 3. Embed (con embedder natural)
                batch_embeds = self._embed_tokens(batch_tokens)

                # 4. Forward Transformer (encoder-only, NO decoder input)
                outputs_norm = self.model(batch_embeds)

                # 5. RevIN denormalize
                outputs = self.revin(outputs_norm, mode='denorm')

                # 6. Calcular loss
                # Extraer predicciones (últimos pred_len timesteps)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        """
        Loop de entrenamiento con early stopping.

        Args:
            setting: String identificador del experimento

        Returns:
            Modelo entrenado
        """
        # Cargar datos
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Directorio de checkpoints
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # === LOOP DE EPOCHS ===
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                # Mover a dispositivo
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # === PIPELINE PLAN A ===

                # 1. RevIN normalize
                batch_x_norm = self.revin(batch_x, mode='norm')

                # 2. Tokenizar
                batch_tokens = self._tokenize_batch(batch_x_norm)

                # 3. Embed
                batch_embeds = self._embed_tokens(batch_tokens)

                # 4. Forward Transformer
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs_norm = self.model(batch_embeds)
                        # 5. RevIN denormalize
                        outputs = self.revin(outputs_norm, mode='denorm')
                        # 6. Loss
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs_norm = self.model(batch_embeds)
                    outputs = self.revin(outputs_norm, mode='denorm')
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # Logging
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                # Backward
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # Fin de epoch
            print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch+1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # Cargar mejor modelo
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, weights_only=True))

        return self.model

    def test(self, setting, test=0):
        """
        Evalúa modelo en test y guarda métricas.

        Args:
            setting: Identificador del experimento
            test: Si 1, carga modelo desde checkpoint
        """
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), weights_only=True))

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

                # === PIPELINE PLAN A ===
                batch_x_norm = self.revin(batch_x, mode='norm')
                batch_tokens = self._tokenize_batch(batch_x_norm)
                batch_embeds = self._embed_tokens(batch_tokens)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs_norm = self.model(batch_embeds)
                else:
                    outputs_norm = self.model(batch_embeds)

                outputs = self.revin(outputs_norm, mode='denorm')

                # Extraer predicciones
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # Inverse transform si hay escalado adicional
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                preds.append(outputs)
                trues.append(batch_y)

                # Visualizar cada 20 muestras para quality check
                if i % 20 == 0:
                    # Copias para no modificar arrays originales
                    input_viz = batch_x.detach().cpu().numpy()
                    output_viz = outputs.copy()
                    true_viz = batch_y.copy()

                    # Desnormalizar: convertir de escala normalizada a escala original
                    # para interpretabilidad en gráficos
                    if test_data.scale and self.args.inverse:
                        shape = input_viz.shape
                        input_viz = test_data.inverse_transform(input_viz.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        shape = output_viz.shape
                        output_viz = test_data.inverse_transform(output_viz.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        shape = true_viz.shape
                        true_viz = test_data.inverse_transform(true_viz.reshape(shape[0] * shape[1], -1)).reshape(shape)

                    # Concatenar input + output para mostrar contexto completo en gráfico
                    # (input histórico + predicción futura)
                    gt = np.concatenate((input_viz[0, :, -1], true_viz[0, :, -1]), axis=0)
                    pd = np.concatenate((input_viz[0, :, -1], output_viz[0, :, -1]), axis=0)

                    # Métricas de esta muestra individual para mostrar en título
                    sample_mse = np.mean((output_viz[0, :, -1] - true_viz[0, :, -1]) ** 2)
                    sample_mae = np.mean(np.abs(output_viz[0, :, -1] - true_viz[0, :, -1]))

                    # Guardar gráfico con métricas
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'),
                          mse=sample_mse, mae=sample_mae)

        # Concatenar predicciones
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # Guardar resultados
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Métricas
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'Technique: {self.technique}')
        print(f'mse:{mse}, mae:{mae}')

        # Guardar en archivo de resultados
        f = open("result_plan_a.txt", 'a')
        f.write(setting + "  \n")
        f.write(f'Technique: {self.technique}\n')
        f.write(f'mse:{mse}, mae:{mae}\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
