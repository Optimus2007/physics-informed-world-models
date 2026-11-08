"""Acrobot swing-up (torque at joint 2 only), point masses at link ends. theta1=0 hangs DOWN.

Positions: p1 = l1 (sin t1, -cos t1);  p2 = p1 + l2 (sin(t1+t2), -cos(t1+t2)), t2 relative.
Manipulator form  M(q) qdd + C(q,qd) + G(q) = [0, tau] - b qd:
  M11 = (m1+m2) l1^2 + m2 l2^2 + 2 m2 l1 l2 cos t2,  M12 = m2 l2^2 + m2 l1 l2 cos t2,  M22 = m2 l2^2
  h = m2 l1 l2 sin t2;  C = [-h(2 qd1 qd2 + qd2^2), h qd1^2]
  G1 = (m1+m2) g l1 sin t1 + m2 g l2 sin(t1+t2),  G2 = m2 g l2 sin(t1+t2)
V = -(m1+m2) g l1 cos t1 - m2 g l2 cos(t1+t2). State (t1, t2, qd1, qd2).
"""
import math
import numpy as np
import torch
from .base import Sim, DTYPE


def acrobot_deriv(s, u, m1, m2, l1, l2, g, b):
    t1, t2, w1, w2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    c2, s2 = torch.cos(t2), torch.sin(t2)
    M11 = (m1 + m2) * l1 ** 2 + m2 * l2 ** 2 + 2 * m2 * l1 * l2 * c2
    M12 = m2 * l2 ** 2 + m2 * l1 * l2 * c2
    M22 = m2 * l2 ** 2 + 0 * c2
    h = m2 * l1 * l2 * s2
    C1 = -h * (2 * w1 * w2 + w2 ** 2)
    C2 = h * w1 ** 2
    G1 = (m1 + m2) * g * l1 * torch.sin(t1) + m2 * g * l2 * torch.sin(t1 + t2)
    G2 = m2 * g * l2 * torch.sin(t1 + t2)
    r1 = -C1 - G1 - b * w1
    r2 = u[..., 0] - C2 - G2 - b * w2
    det = M11 * M22 - M12 ** 2
    a1 = (M22 * r1 - M12 * r2) / det
    a2 = (-M12 * r1 + M11 * r2) / det
    return torch.stack([w1, w2, a1, a2], -1)


def acrobot_deriv_np(s, u, m1, m2, l1, l2, g, b):
    d = acrobot_deriv(torch.tensor(s)[None], torch.tensor([[u]], dtype=torch.float64),
                      m1, m2, l1, l2, g, b)
    return d[0].numpy()


class Acrobot(Sim):
    name = "acrobot"
    state_dim = 4
    angle_idx = (0, 1)
    default_params = dict(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.8, b=0.0, umax=5.0)
    dt = 0.05
    substeps = 8

    def deriv(self, s, u, params=None):
        p = params or self.params
        return acrobot_deriv(s, u, p["m1"], p["m2"], p["l1"], p["l2"], p["g"], p["b"])

    def energy(self, s):
        p = self.params
        t1, t2, w1, w2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        c2 = torch.cos(t2)
        M11 = (p["m1"] + p["m2"]) * p["l1"] ** 2 + p["m2"] * p["l2"] ** 2 + 2 * p["m2"] * p["l1"] * p["l2"] * c2
        M12 = p["m2"] * p["l2"] ** 2 + p["m2"] * p["l1"] * p["l2"] * c2
        M22 = p["m2"] * p["l2"] ** 2
        T = 0.5 * (M11 * w1 ** 2 + 2 * M12 * w1 * w2 + M22 * w2 ** 2)
        V = -(p["m1"] + p["m2"]) * p["g"] * p["l1"] * torch.cos(t1) - p["m2"] * p["g"] * p["l2"] * torch.cos(t1 + t2)
        return T + V

    def tip_height(self, s):
        p = self.params
        return -p["l1"] * torch.cos(s[..., 0]) - p["l2"] * torch.cos(s[..., 0] + s[..., 1])

    def reward(self, s, u):
        top = self.params["l1"] + self.params["l2"]
        return -((top - self.tip_height(s)) ** 2 + 0.05 * (s[..., 2] ** 2 + s[..., 3] ** 2)
                 + 0.001 * u[..., 0] ** 2)

    def sample_init(self, n, gen=None):
        r = lambda: torch.rand(n, generator=gen, dtype=DTYPE) * 2 - 1
        return torch.stack([0.1 * r(), 0.1 * r(), 0.1 * r(), 0.1 * r()], -1)

    def sample_init_wide(self, n, gen=None):
        r = lambda: torch.rand(n, generator=gen, dtype=DTYPE) * 2 - 1
        return torch.stack([math.pi * r(), math.pi * r(), 2 * r(), 2 * r()], -1)
