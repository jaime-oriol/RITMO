"""
Familia de mecanismos de Self-Attention para series temporales.
Incluye: FullAttention, ProbAttention (Informer), DSAttention, Reformer, TSA.
Cada variante optimiza para diferentes escenarios (velocidad, memoria, etc.).
"""

import torch  # Framework de deep learning
import torch.nn as nn  # Módulo de redes neuronales
import numpy as np  # Operaciones numéricas
from math import sqrt  # Raíz cuadrada
from utils.masking import TriangularCausalMask, ProbMask  # Máscaras de atención
from reformer_pytorch import LSHSelfAttention  # Atención eficiente LSH
from einops import rearrange, repeat  # Manipulación de tensores


class DSAttention(nn.Module):
    """
    De-Stationary Attention (Non-stationary Transformer).
    Añade factores tau y delta para manejar series no estacionarias.
    tau: escala las puntuaciones, delta: sesgo aditivo.
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag  # ¿Usar máscara causal?
        self.output_attention = output_attention  # ¿Devolver pesos de atención?
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        queries: [B, L, H, E] - consultas
        keys: [B, S, H, E] - claves
        values: [B, S, H, D] - valores
        tau: factor de escala aprendido
        delta: sesgo aditivo aprendido
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)  # Escala 1/sqrt(d_k)

        # Preparar factores de-stationarity
        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)

        # Atencion con factores de-stationarity
        # scores = Q·K^T * tau + delta
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        # Aplicar máscara causal si es necesario
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Softmax + dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # Multiplicar por valores
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    """
    Atención completa estándar (Vaswani et al., 2017).
    Complejidad O(L²) - calcula todas las interacciones.
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """Atención estándar: softmax(Q·K^T / sqrt(d)) · V"""
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Producto escalar Q·K^T
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # Máscara causal (para autoregresivo)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Softmax normalizado + dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # Salida: suma ponderada de valores
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    """
    ProbSparse Attention (Informer).
    Selecciona solo las queries más informativas para reducir O(L²) a O(L·log(L)).
    Usa muestreo para identificar queries con alta "sparsity".
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor  # Factor de muestreo c
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Selecciona las top-n queries más informativas.
        Mide "sparsity" como max(Q·K) - mean(Q·K).
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Muestrear aleatoriamente sample_k claves
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        # Calcular Q·K para muestras
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # Medir sparsity: M = max - mean
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        # Seleccionar top-n queries
        M_top = M.topk(n_top, sorted=False)[1]

        # Calcular atención solo para queries seleccionadas
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """Contexto inicial: media de valores o acumulado (para causal)."""
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert (L_Q == L_V)  # Solo para self-attention
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """Actualiza contexto solo para queries seleccionadas."""
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        # Actualizar solo posiciones seleccionadas
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """ProbSparse attention: O(L·log(L)) en vez de O(L²)."""
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # Calcular número de muestras
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        # Seleccionar queries más informativas
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # Escalar puntuaciones
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # Calcular y actualizar contexto
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    """
    Capa de atención completa con proyecciones Q, K, V.
    Envuelve cualquier mecanismo de atención (Full, Prob, DS, etc.).
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention  # Mecanismo de atención interno
        # Proyecciones lineales para Q, K, V
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        # Proyección de salida
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        queries, keys, values: [B, L, d_model]
        Salida: [B, L, d_model]
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Proyectar y dividir en cabezas
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Aplicar atención
        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        # Concatenar cabezas y proyectar
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    """
    Atención LSH (Locality Sensitive Hashing) de Reformer.
    Agrupa queries similares en buckets para reducir complejidad.
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        # LSH Self-Attention de reformer-pytorch
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        """Ajusta longitud para que sea divisible por bucket_size*2."""
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        """En Reformer: queries = keys (self-attention)."""
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    """
    Two Stage Attention (TSA) de Crossformer.
    Etapa 1: Atención temporal (entre timesteps)
    Etapa 2: Atención dimensional (entre variables)
    Usa un "router" learnable para agregar información entre dimensiones.
    """

    def __init__(self, configs, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        # Atención temporal: entre segmentos de tiempo
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=False), d_model, n_heads)
        # Atención dimensional: sender y receiver
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=False), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=False), d_model, n_heads)
        # Router: vectores aprendibles para comunicación entre dimensiones
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)
        # Normalizaciones
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        # MLPs
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        x: [B, D, L, d_model] donde D=variables, L=segmentos
        """
        batch = x.shape[0]

        # Etapa 1: atencion temporal
        # Reorganizar para atención entre segmentos temporales
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(time_in, time_in, time_in, attn_mask=None, tau=None, delta=None)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Etapa 2: atencion dimensional
        # Reorganizar para atención entre variables
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        # Expandir router para el batch
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)

        # Sender: router recibe de variables
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        # Receiver: variables reciben del router
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)

        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        # Volver a forma original
        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out
