"""Forward-Backward HMM en PyTorch GPU.

HMMForwardBackwardGPU pre-cachea params + constantes de emisión en GPU al init.
Acepta torch.Tensor directamente — cero copias CPU↔GPU por batch.
log_const e inv_2sig2 pre-computados una sola vez.
"""
import numpy as np
import torch


def _forward_gpu_fn(log_B: torch.Tensor, log_A: torch.Tensor, log_pi: torch.Tensor) -> torch.Tensor:
    """
    log_B  : [N, T, K]
    log_A  : [K, K]
    log_pi : [K]
    returns: log_alpha_all [N, T, K]
    """
    T = log_B.shape[1]
    log_alpha_all = torch.empty_like(log_B)
    log_alpha = log_pi.unsqueeze(0) + log_B[:, 0, :]          # [N, K]
    log_alpha_all[:, 0, :] = log_alpha
    for t in range(1, T):
        trans = log_alpha.unsqueeze(2) + log_A.unsqueeze(0)   # [N, K, K]
        log_alpha = torch.logsumexp(trans, dim=1) + log_B[:, t, :]
        log_alpha_all[:, t, :] = log_alpha
    return log_alpha_all


def _backward_gamma_gpu_fn(
    log_alpha_all: torch.Tensor,
    log_B: torch.Tensor,
    log_A: torch.Tensor,
) -> torch.Tensor:
    """
    log_alpha_all : [N, T, K]
    log_B         : [N, T, K]
    log_A         : [K, K]
    returns       : gamma [N, T, K]
    """
    N, T, K = log_alpha_all.shape
    gamma = torch.empty_like(log_alpha_all)
    log_beta = torch.zeros(N, K, dtype=log_alpha_all.dtype, device=log_alpha_all.device)
    gamma[:, T - 1, :] = torch.softmax(log_alpha_all[:, T - 1, :] + log_beta, dim=-1)
    for t in range(T - 2, -1, -1):
        inner = log_A.unsqueeze(0) + log_B[:, t + 1, :].unsqueeze(1) + log_beta.unsqueeze(1)
        log_beta = torch.logsumexp(inner, dim=2)
        gamma[:, t, :] = torch.softmax(log_alpha_all[:, t, :] + log_beta, dim=-1)
    return gamma


# ─────────────────────────────────────────────────────────────────────────────
# Clase stateful — params y constantes pre-cacheados en GPU al init
# ─────────────────────────────────────────────────────────────────────────────

class HMMForwardBackwardGPU(torch.nn.Module):
    """Forward-backward GPU con params pre-cacheados. Cero copias CPU↔GPU por batch.

    Uso:
        fb = HMMForwardBackwardGPU(hmm_params, device='cuda')
        # obs: [N, T] torch.Tensor ya en device
        gamma = fb(obs)   # [N, T, K] torch.Tensor en device
    """

    def __init__(self, hmm_params: dict, device: str = 'cuda') -> None:
        super().__init__()
        EPS = 1e-20
        A     = np.asarray(hmm_params['A'],     dtype=np.float32)
        pi    = np.asarray(hmm_params['pi'],    dtype=np.float32)
        mu    = np.asarray(hmm_params['mu'],    dtype=np.float32)
        sigma = np.asarray(hmm_params['sigma'], dtype=np.float32)

        sig_sq = sigma ** 2 + 1e-8

        # Params log en GPU — registrados como buffers (se mueven con .to(device))
        self.register_buffer('log_A',     torch.log(torch.from_numpy(A  + EPS)))
        self.register_buffer('log_pi',    torch.log(torch.from_numpy(pi + EPS)))
        self.register_buffer('mu_t',      torch.from_numpy(mu))
        # Constantes de emisión — pre-computadas, nunca se recalculan
        self.register_buffer('log_c',     torch.from_numpy(
            (-0.5 * np.log(2.0 * np.pi * sig_sq)).astype(np.float32)))
        self.register_buffer('inv_2sig2', torch.from_numpy(
            (0.5 / sig_sq).astype(np.float32)))

        self.to(device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs   : [N, T] float32 torch.Tensor en device
        return: gamma [N, T, K] float32 en device
        """
        diff  = obs.unsqueeze(-1) - self.mu_t              # [N, T, K]
        log_B = self.log_c - diff * diff * self.inv_2sig2  # [N, T, K]
        log_alpha_all = _forward_gpu_fn(log_B, self.log_A, self.log_pi)
        return _backward_gamma_gpu_fn(log_alpha_all, log_B, self.log_A)


# ─────────────────────────────────────────────────────────────────────────────
# API de compatibilidad — para código que aún llama la función directamente
# ─────────────────────────────────────────────────────────────────────────────

def forward_backward_batch_torch(
    observations: np.ndarray,
    A: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    device: str = 'cuda',
):
    """Wrapper numpy-compatible. Preferir HMMForwardBackwardGPU en loops de entrenamiento."""
    EPS = 1e-20
    sig_sq = sigma.astype(np.float32) ** 2 + 1e-8

    obs    = torch.tensor(observations, dtype=torch.float32, device=device)
    log_A  = torch.log(torch.tensor(A  + EPS, dtype=torch.float32, device=device))
    log_pi = torch.log(torch.tensor(pi + EPS, dtype=torch.float32, device=device))
    mu_t   = torch.tensor(mu,    dtype=torch.float32, device=device)
    log_c  = torch.tensor((-0.5 * np.log(2.0 * np.pi * sig_sq)).astype(np.float32), device=device)
    i2s2   = torch.tensor((0.5 / sig_sq).astype(np.float32), device=device)

    diff  = obs.unsqueeze(-1) - mu_t
    log_B = log_c - diff * diff * i2s2

    log_alpha_all = _forward_gpu_fn(log_B, log_A, log_pi)
    gamma = _backward_gamma_gpu_fn(log_alpha_all, log_B, log_A)
    return gamma, None, None
