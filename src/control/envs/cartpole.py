"""Cart-pole swing-up (point mass m on massless rod l, cart mass M), theta=0 upright.

Lagrangian (D1 in docs/DECISIONS.md):
  T = 1/2 (M+m) xd^2 + m l xd thd cos(th) + 1/2 m l^2 thd^2,   V = m g l cos(th)
  EL equations (force F on cart, viscous cart friction b xd):
    (M+m) xdd + m l cos(th) thdd - m l sin(th) thd^2 = F - b xd
    m l cos(th) xdd + m l^2 thdd - m g l sin(th)     = 0
  Solve: xdd = (F - b xd + m l sin(th) thd^2 - m g sin(th) cos(th)) / (M + m sin^2 th)
         thdd = (g sin(th) - cos(th) xdd) / l
State (x, th, xd, thd).
"""
import math
import numpy as np
import torch
from .base import Sim, DTYPE, angle_normalize


def cartpole_deriv(s, u, M, m, l, g, b):
    th, xd, thd = s[..., 1], s[..., 2], s[..., 3]
    sn, cs = torch.sin(th), torch.cos(th)
    xdd = (u[..., 0] - b * xd + m * l * sn * thd ** 2 - m * g * sn * cs) / (M + m * sn ** 2)
    thdd = (g * sn - cs * xdd) / l
    return torch.stack([xd, thd, xdd, thdd], -1)


def cartpole_deriv_np(s, u, M, m, l, g, b):
    x, th, xd, thd = s
    sn, cs = np.sin(th), np.cos(th)
    xdd = (u - b * xd + m * l * sn * thd ** 2 - m * g * sn * cs) / (M + m * sn ** 2)
    return np.array([xd, thd, xdd, (g * sn - cs * xdd) / l])


class CartPole(Sim):
    name = "cartpole"
    state_dim = 4
    default_params = dict(M=1.0, m=0.3, l=0.5, g=9.8, b=0.0, umax=10.0)
    dt = 0.05
    substeps = 4

    def deriv(self, s, u, params=None):
        p = params or self.params
        return cartpole_deriv(s, u, p["M"], p["m"], p["l"], p["g"], p["b"])

    def energy(self, s):
        p = self.params
        x, th, xd, thd = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        T = 0.5 * (p["M"] + p["m"]) * xd ** 2 + p["m"] * p["l"] * xd * thd * torch.cos(th) \
            + 0.5 * p["m"] * p["l"] ** 2 * thd ** 2
        return T + p["m"] * p["g"] * p["l"] * torch.cos(th)

    def reward(self, s, u):
        th = angle_normalize(s[..., 1])
        return -(th ** 2 + 0.1 * s[..., 3] ** 2 + 0.1 * s[..., 0] ** 2 + 0.01 * s[..., 2] ** 2
                 + 0.001 * u[..., 0] ** 2)

    def sample_init(self, n, gen=None):
        """Hanging down (th ~ pi) with small noise, for swing-up."""
        r = lambda: torch.rand(n, generator=gen, dtype=DTYPE) * 2 - 1
        return torch.stack([0.1 * r(), math.pi + 0.1 * r(), 0.1 * r(), 0.1 * r()], -1)

    def sample_init_wide(self, n, gen=None):
        r = lambda: torch.rand(n, generator=gen, dtype=DTYPE) * 2 - 1
        return torch.stack([r(), math.pi * r(), r(), 4 * r()], -1)
