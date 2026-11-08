"""PETS-style MLP ensemble (Chua et al. 2018), simplified: K members predict normalised state increments,
trained with MSE on bootstrap-weighted data (NOT Gaussian NLL; logged in DECISIONS D4). Planning uses the
ensemble MEAN (deterministic); the spread is available via `member_preds` for later uncertainty work."""
import torch
import torch.nn as nn
from .base import WorldModel, DT


class EnsembleLinear(nn.Module):
    def __init__(self, K, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.randn(K, i, o, dtype=DT) / i ** 0.5)
        self.b = nn.Parameter(torch.zeros(K, 1, o, dtype=DT))

    def forward(self, x):  # x:(K,B,i)
        return torch.baddbmm(self.b, x, self.w)


class MLPEnsemble(WorldModel):
    name = "mlp_ensemble"

    def __init__(self, env, K=5, hidden=(200, 200, 200), n_data=1):
        super().__init__(env)
        self.K = K
        dims = [self.nfeat + self.m] + list(hidden) + [self.n]
        self.layers = nn.ModuleList(EnsembleLinear(K, a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.register_buffer("fmean", torch.zeros(self.nfeat, dtype=DT))
        self.register_buffer("fstd", torch.ones(self.nfeat, dtype=DT))
        self.boot = None  # (K, N) bootstrap weights, set in fit_prepare

    def set_norm(self, x, xn):
        super().set_norm(x, xn)
        f = self.features(x); self.fmean.copy_(f.mean(0)); self.fstd.copy_(f.std(0).clamp_min(1e-3))

    def fit_prepare(self, N, gen):
        self.boot = torch.poisson(torch.ones(self.K, N, dtype=DT), generator=gen)

    def member_preds(self, x, u):
        """-> (K,B,n) next states, one per member."""
        f = torch.cat([(self.features(x) - self.fmean) / self.fstd, u / self.umax], -1)
        h = f[None].expand(self.K, -1, -1)
        for i, l in enumerate(self.layers):
            h = l(h)
            if i < len(self.layers) - 1:
                h = torch.nn.functional.silu(h)
        return x[None] + self.dmean + h * self.dstd

    def step(self, x, u):
        return self.member_preds(x, u).mean(0)

    def loss(self, x, u, xn, idx=None):
        err = (((self.member_preds(x, u) - xn[None]) / self.dstd) ** 2).mean(-1)   # (K,B)
        w = self.boot[:, idx] if (self.boot is not None and idx is not None) else torch.ones_like(err)
        return ((w * err).sum(1) / w.sum(1).clamp_min(1)).sum()  # sum over members (independent training)
