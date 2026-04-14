"""
Experimento Plan B: HMM+Transformer (multivariate M) vs 4 baselines SOTA.

Usa channel-independence estilo PatchTST:
    [B, L, C] --permute+reshape--> [B*C, L, 1] --pipeline HMM+Transformer-->
    [B*C, pred_len, 1] --reshape--> [B, pred_len, C]

Un unico HMM compartido por dataset (entrenado sobre los C canales apilados como
secuencias independientes con baum_welch_batch). Cache en cache/hmm_M_{dataset}_K{k}.pth.

Solo soporta las variantes HMM (hmm_soft, hmm_soft_residual, hmm, hmm_augmented,
hmm_split, hmm_patched). Los 4 baselines SOTA se corren con sus propios scripts
originales via exp_long_term_forecasting.
"""

import copy
import gc
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from layers.StandardNorm import Normalize
from hmm import viterbi_batch, forward_backward_batch
from embeddings import EmbeddingGenerator

warnings.filterwarnings('ignore')

_VALID_TECHNIQUES = (
    'hmm', 'hmm_soft', 'hmm_soft_residual',
    'hmm_augmented', 'hmm_split', 'hmm_patched',
)


class Exp_Plan_B(Exp_Basic):
    """
    Plan B: HMM multivariate (channel-independence) + TransformerCommon vs 4 baselines.

    Diferencias clave respecto a Exp_Plan_A:
        - args.features == 'M' (todas las variables entran, todas salen).
        - RevIN per-canal con num_features=C.
        - HMM compartido entre canales (cache hmm_M_*).
        - Channel-independence en tokenizacion y transformer.
        - Loss y metricas sobre los C canales (consistente con leaderboard TSLib).
    """

    def __init__(self, args):
        if args.features != 'M':
            raise ValueError(
                f"Exp_Plan_B requiere --features M (recibido: {args.features}). "
                "El Plan B compara HMM-M contra baselines SOTA en multivariate."
            )
        if not hasattr(args, 'technique') or args.technique not in _VALID_TECHNIQUES:
            raise ValueError(
                f"args.technique debe ser una de {_VALID_TECHNIQUES}"
            )

        super(Exp_Plan_B, self).__init__(args)
        self.technique = args.technique
        self.n_channels = args.enc_in  # C canales del dataset original

        # RevIN per-canal (num_features=C), externa y sin parametros aprendibles
        self.revin = Normalize(
            num_features=self.n_channels,
            eps=1e-5,
            affine=False,
            subtract_last=False,
            non_norm=False,
        )

        self._init_embedder()

    # ------------------------------------------------------------------
    # Model / data / optim plumbing
    # ------------------------------------------------------------------
    def _build_model(self):
        """TransformerCommon construido con enc_in=1 (channel-independence)."""
        cfg = copy.copy(self.args)
        cfg.enc_in = 1  # el transformer ve un unico canal por elemento de batch
        model = self.model_dict['TransformerCommon'].Model(cfg).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _init_embedder(self):
        d_model = self.args.d_model
        hmm_k = getattr(self.args, 'hmm_k', 5)
        # Preferir --hmm_cache_path explicito; en caso contrario, derivar desde args.data.
        # IMPORTANTE: cuando se usa --data custom (Weather/ECL) el path por defecto es
        # ambiguo; pasar siempre --hmm_cache_path explicito en Plan B.
        dataset_tag = self.args.data.lower()
        default_cache = f'./cache/hmm_M_{dataset_tag}_K{hmm_k}.pth'
        cache_path = getattr(self.args, 'hmm_cache_path', '') or default_cache
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Cache HMM-M no encontrado: {cache_path}. "
                "Entrenar primero con scripts/plan_b/train_hmm_caches.py o la Celda 2 "
                "de notebooks/k_sweep_plan_b.ipynb."
            )

        hmm_params = torch.load(cache_path, weights_only=False)
        self.embedder = EmbeddingGenerator(
            hmm_params=hmm_params, d_model=d_model, device=self.device,
        )
        self.hmm_params = hmm_params
        # Arrays numpy para las rutinas batch de forward-backward / viterbi
        self._hmm_A = self._as_numpy(hmm_params['A'])
        self._hmm_pi = self._as_numpy(hmm_params['pi'])
        self._hmm_mu = self._as_numpy(hmm_params['mu'])
        self._hmm_sigma = self._as_numpy(hmm_params['sigma'])
        self._hmm_cache_path = cache_path

    @staticmethod
    def _as_numpy(x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    def _select_optimizer(self):
        params = list(self.model.parameters())
        if self.technique in ('hmm', 'hmm_soft'):
            params += list(self.embedder.projection.parameters())
        elif self.technique == 'hmm_soft_residual':
            params += list(self.embedder.projection_residual.parameters())
        elif self.technique == 'hmm_augmented':
            params += list(self.embedder.projection_augmented.parameters())
        elif self.technique == 'hmm_patched':
            params += list(self.embedder.projection_patched.parameters())
        elif self.technique == 'hmm_split':
            params += list(self.embedder.projection_value.parameters())
            params += list(self.embedder.projection_gamma.parameters())
            params += list(self.embedder.norm_split.parameters())
        return optim.Adam(params, lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    # ------------------------------------------------------------------
    # Pipeline core (channel-independence)
    # ------------------------------------------------------------------
    def _flatten_channels(self, batch_x_norm):
        """[B, L, C] -> [B*C, L] (numpy) preservando orden row-major por canal."""
        # permute a [B, C, L] -> [B*C, L]
        B, L, C = batch_x_norm.shape
        flat = batch_x_norm.permute(0, 2, 1).contiguous().view(B * C, L)
        return flat.detach().cpu().numpy()

    def _tokenize_embed_flat(self, batch_x_norm):
        """Tokeniza + embed con channel-independence.

        Devuelve embeddings [B*C, seq_len_out, d_model] listos para TransformerCommon.
        """
        B, L, C = batch_x_norm.shape
        flat_np = self._flatten_channels(batch_x_norm)  # [B*C, L]

        if self.technique == 'hmm':
            states = viterbi_batch(
                flat_np, self._hmm_A, self._hmm_pi, self._hmm_mu, self._hmm_sigma,
            )  # [B*C, L]
            states_t = torch.from_numpy(states).long().to(self.device)
            raw = self.embedder.embedding_table[states_t]  # [B*C, L, 2+K]
            return self.embedder.projection(raw)

        # Rutas gamma-based
        gamma, _, _ = forward_backward_batch(
            flat_np, self._hmm_A, self._hmm_pi, self._hmm_mu, self._hmm_sigma,
        )  # [B*C, L, K]
        gamma_t = torch.from_numpy(gamma).float().to(self.device)

        if self.technique == 'hmm_soft':
            raw = torch.matmul(gamma_t, self.embedder.embedding_table)
            return self.embedder.projection(raw)

        # Tecnicas que requieren x crudo normalizado ademas de gamma
        x_t = torch.from_numpy(flat_np).float().to(self.device)  # [B*C, L]

        if self.technique == 'hmm_soft_residual':
            mu_soft = torch.matmul(gamma_t, self.embedder.mu)
            sigma_soft = torch.clamp(torch.matmul(gamma_t, self.embedder.sigma), min=1e-6)
            r_t = (x_t - mu_soft) / sigma_soft
            A_soft = torch.matmul(gamma_t, self.embedder.embedding_table[:, 2:])
            features = torch.cat([
                r_t.unsqueeze(-1), mu_soft.unsqueeze(-1),
                sigma_soft.unsqueeze(-1), A_soft,
            ], dim=-1)
            return self.embedder.projection_residual(features)

        if self.technique == 'hmm_augmented':
            features = torch.cat([x_t.unsqueeze(-1), gamma_t], dim=-1)
            return self.embedder.projection_augmented(features)

        if self.technique == 'hmm_split':
            emb_v = self.embedder.projection_value(x_t.unsqueeze(-1))
            emb_g = self.embedder.projection_gamma(gamma_t)
            return self.embedder.norm_split(torch.cat([emb_v, emb_g], dim=-1))

        if self.technique == 'hmm_patched':
            features = torch.cat([x_t.unsqueeze(-1), gamma_t], dim=-1)
            BC, T, Fdim = features.shape
            P = self.embedder.patch_len_hmm
            num_patches = T // P
            features = features[:, :num_patches * P, :]
            patches = features.reshape(BC, num_patches, P * Fdim)
            embeds = self.embedder.projection_patched(patches)
            embeds = embeds.transpose(1, 2)
            embeds = F.adaptive_avg_pool1d(embeds, self.args.seq_len)
            return embeds.transpose(1, 2)

        raise ValueError(f"Tecnica HMM desconocida: {self.technique}")

    def _forward_pipeline(self, batch_x):
        """RevIN -> tokenizar/embed -> transformer (channel-indep) -> reshape -> denorm.

        Args:
            batch_x: [B, L, C] datos crudos.

        Returns:
            outputs: [B, pred_len, C] predicciones en escala original.
        """
        B, L, C = batch_x.shape
        batch_x_norm = self.revin(batch_x, mode='norm')  # [B, L, C]

        embeds = self._tokenize_embed_flat(batch_x_norm)  # [B*C, seq_len_eff, d_model]
        preds_flat = self.model(embeds)  # [B*C, pred_len, 1]
        preds_flat = preds_flat.squeeze(-1)  # [B*C, pred_len]
        preds = preds_flat.view(B, C, self.args.pred_len).permute(0, 2, 1).contiguous()
        # [B, pred_len, C]
        outputs = self.revin(preds, mode='denorm')
        return outputs

    # ------------------------------------------------------------------
    # Train / vali / test
    # ------------------------------------------------------------------
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y, _, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self._forward_pipeline(batch_x)

                f_dim = 0  # Plan B siempre M -> todos los canales
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                total_loss.append(criterion(outputs, batch_y).item())
        self.model.train()
        return float(np.average(total_loss))

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        use_amp = getattr(self.args, 'use_amp', False) and self.args.use_gpu
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_start = time.time()

            for i, (batch_x, batch_y, _, _) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_pipeline(batch_x)
                        outputs = outputs[:, -self.args.pred_len:, :]
                        batch_y = batch_y[:, -self.args.pred_len:, :]
                        loss = criterion(outputs, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self._forward_pipeline(batch_x)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :]
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")

            train_loss = float(np.average(train_loss))
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print(
                f"Epoch: {epoch+1}, Steps: {train_steps} | "
                f"Train: {train_loss:.7f} Vali: {vali_loss:.7f} Test: {test_loss:.7f} "
                f"({time.time()-epoch_start:.1f}s)"
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, weights_only=True))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(
                os.path.join('./checkpoints/' + setting, 'checkpoint.pth'),
                weights_only=True,
            ))

        preds, trues = [], []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self._forward_pipeline(batch_x)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]

                outputs_np = outputs.detach().cpu().numpy()
                true_np = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = true_np.shape
                    outputs_np = test_data.inverse_transform(
                        outputs_np.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                    true_np = test_data.inverse_transform(
                        true_np.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)

                preds.append(outputs_np)
                trues.append(true_np)

                if i % 20 == 0:
                    gt = np.concatenate(
                        (batch_x[0, :, -1].detach().cpu().numpy(), true_np[0, :, -1]),
                        axis=0,
                    )
                    pd = np.concatenate(
                        (batch_x[0, :, -1].detach().cpu().numpy(), outputs_np[0, :, -1]),
                        axis=0,
                    )
                    sample_mse = float(np.mean((outputs_np[0, :, -1] - true_np[0, :, -1]) ** 2))
                    sample_mae = float(np.mean(np.abs(outputs_np[0, :, -1] - true_np[0, :, -1])))
                    visual(gt, pd, os.path.join(folder_path, f'{i}.pdf'),
                           mse=sample_mse, mae=sample_mae)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'Technique: {self.technique} | K={getattr(self.args, "hmm_k", "?")}')
        print(f'mse:{mse}, mae:{mae}')

        with open('result_plan_b.txt', 'a') as f:
            f.write(setting + '\n')
            f.write(f'Technique: {self.technique} | K={getattr(self.args, "hmm_k", "?")}\n')
            f.write(f'mse:{mse}, mae:{mae}\n\n')

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # Liberar memoria al cerrar experimento (WSL safety)
        del preds, trues
        gc.collect()
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
