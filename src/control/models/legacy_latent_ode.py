"""`legacy_latent_ode`: the existing PhysicsInformedWorldModel (src/models.py: encoder -> dopri5 neural ODE -> decoder)
adapted to accept actions so it can be compared fairly as an extra baseline.

What is kept: tanh MLP encoder/decoder/ODE function (hidden 64, latent 4), torchdiffeq dopri5.
What had to change (DECISIONS D5): (1) the ODE is dz/dt = f(z, u) with u held constant over the control interval;
(2) the decoder predicts a RESIDUAL from the current state (x_next = x + dstd*(dec(z1)-dec(z0)) + dmean) because an
absolute-state decoder cannot resolve one-step increments of ~1e-2 relative to the state scale;
(3) the orbit-specific energy-variance penalty is dropped (it hard-codes the 2-body Hamiltonian, and the original
model's 'physics' was only that soft penalty, NOT a Hamiltonian structure).
Attribution: architecture from this repo's src/models.py (MIT, same author)."""
import torch
import torch.nn as nn
from torchdiffeq import odeint
from .base import WorldModel, mlp, DT


class _Func(nn.Module):
    def __init__(self, zdim, m, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(zdim + m, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, zdim)).double()
        self.u = None

    def forward(self, t, z):
        return self.net(torch.cat([z, self.u], -1))


class LegacyLatentODE(WorldModel):
    name = "legacy_latent_ode"

    def __init__(self, env, latent_dim=4, hidden=64):
        super().__init__(env)
        act = nn.Tanh
        self.enc = mlp(self.nfeat, latent_dim, (hidden,), act)
        self.dec = mlp(latent_dim, self.n, (hidden,), act)
        self.func = _Func(latent_dim, self.m, hidden)
        self.t = torch.tensor([0.0, env.dt], dtype=DT)

    def step(self, x, u):
        z0 = self.enc(self.features(x))
        self.func.u = u / self.umax
        z1 = odeint(self.func, z0, self.t, method="dopri5", rtol=1e-5, atol=1e-7)[-1]
        return x + self.dmean + self.dstd * (self.dec(z1) - self.dec(z0))
