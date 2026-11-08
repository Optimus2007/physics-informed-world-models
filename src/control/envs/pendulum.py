"""Pendulum swing-up (uniform rod, pivot at one end), theta=0 upright.

Same equation of motion as Gymnasium Pendulum-v1 (checked in tests/test_envs.py) but with
RK4 integration, adjustable (m,l,g), viscous friction b and torque limit umax.

Lagrangian derivation (see docs/DECISIONS.md D1):
  I = m l^2 / 3,  T = 1/2 I w^2,  V = (m g l / 2) cos(theta)   [theta=0 up => V max]
  L = T - V,  Euler-Lagrange with torque u and viscous friction -b w:
  I th'' = (m g l / 2) sin(theta) + u - b w
  => th'' = (3 g / (2 l)) sin(theta) + 3 u / (m l^2) - 3 b w / (m l^2)
Hamiltonian: p = I w,  H = p^2/(2I) + (m g l/2) cos(theta).
"""
import math
import numpy as np
import torch
from .base import Sim, DTYPE, angle_normalize


def pendulum_deriv(x, u, m, l, g, b):
    """Pure function (params may be python floats or torch tensors => differentiable)."""
    th, w = x[..., 0], x[..., 1]
    acc = 1.5 * g / l * torch.sin(th) + 3.0 * (u[..., 0] - b * w) / (m * l ** 2)
    return torch.stack([w, acc], -1)


def pendulum_deriv_np(x, u, m, l, g, b):
    th, w = x
    return np.array([w, 1.5 * g / l * np.sin(th) + 3.0 * (u - b * w) / (m * l ** 2)])


class Pendulum(Sim):
    name = "pendulum"
    state_dim = 2
    angle_idx = (0,)
    default_params = dict(m=1.0, l=1.0, g=10.0, b=0.0, umax=2.0, usat=1e3)
    dt = 0.05
    substeps = 4

    def clip(self, u):
        """Planner/command range is `umax`; `usat` is a hidden actuator saturation (Phase F 'motor limit')."""
        lim = min(self.params["umax"], self.params["usat"])
        return torch.clamp(u, -lim, lim)

    def deriv(self, x, u, params=None):
        p = params or self.params
        return pendulum_deriv(x, u, p["m"], p["l"], p["g"], p["b"])

    def energy(self, x):
        p = self.params
        I = p["m"] * p["l"] ** 2 / 3
        return 0.5 * I * x[..., 1] ** 2 + 0.5 * p["m"] * p["g"] * p["l"] * torch.cos(x[..., 0])

    def reward(self, x, u):
        """Gymnasium-style cost, returned as reward (<=0). x:(...,2) u:(...,1)."""
        th = angle_normalize(x[..., 0])
        return -(th ** 2 + 0.1 * x[..., 1] ** 2 + 0.001 * u[..., 0] ** 2)

    def sample_init(self, n, gen=None):
        """Uniform theta in [-pi,pi], omega in [-1,1] (as Gymnasium)."""
        th = (torch.rand(n, generator=gen, dtype=DTYPE) * 2 - 1) * math.pi
        w = torch.rand(n, generator=gen, dtype=DTYPE) * 2 - 1
        return torch.stack([th, w], -1)

    def sample_start(self, n, gen=None):
        """Task start distribution: hanging down, small noise (the swing-up problem)."""
        r = lambda: torch.rand(n, generator=gen, dtype=DTYPE) * 2 - 1
        return torch.stack([math.pi + 0.2 * r(), 0.2 * r()], -1)

    def sample_init_wide(self, n, gen=None, wmax=8.0):
        """Wider omega range for data collection (covers high-energy states)."""
        th = (torch.rand(n, generator=gen, dtype=DTYPE) * 2 - 1) * math.pi
        w = (torch.rand(n, generator=gen, dtype=DTYPE) * 2 - 1) * wmax
        return torch.stack([th, w], -1)
