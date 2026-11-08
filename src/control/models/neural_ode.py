"""Neural ODE without physics: dx/dt = f_theta(features(x), u/umax), integrated with RK4 over dt
(action held constant). Isolates 'continuous time' from 'physics'. Same MLP width as the others."""
import torch
import torch.nn as nn
from .base import WorldModel, mlp, rk4, DT


class NeuralODE(WorldModel):
    name = "neural_ode"

    def __init__(self, env, hidden=(200, 200, 200), substeps=2):
        super().__init__(env)
        self.net = mlp(self.nfeat + self.m, self.n, hidden)
        self.sub = substeps
        self.register_buffer("fscale", torch.ones(self.n, dtype=DT))

    def set_norm(self, x, xn):
        super().set_norm(x, xn); self.fscale.copy_(self.dstd / self.dt)

    def f(self, x, u):
        return self.net(torch.cat([self.features(x), u / self.umax], -1)) * self.fscale

    def step(self, x, u):
        return rk4(self.f, x, u, self.dt, self.sub)
