"""Small neural residual added to a physics vector field (APHYNITY-flavoured, Yin et al. 2021):
   dx/dt = f_phys(x,u; params) + f_res(features(x), u).
Output layer is initialised ~0 so training starts from the pure physics model. A penalty on ||f_res||^2
(weight `res_l2` in physics_structured) prefers explaining dynamics with physics where possible."""
import torch
import torch.nn as nn
from .base import mlp, DT


class Residual(nn.Module):
    def __init__(self, nfeat, m, n, hidden=(64, 64), scale=1.0):
        super().__init__()
        self.net = mlp(nfeat + m, n, hidden, last_scale=1e-3)
        self.scale = scale   # state-derivative units per net output unit

    def forward(self, feat, u_norm, fscale):
        return self.net(torch.cat([feat, u_norm], -1)) * fscale
