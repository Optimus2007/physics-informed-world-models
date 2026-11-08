"""Structured physics model: the analytic Lagrangian equations of motion of the env with LEARNABLE physical
parameters (positive via exp / softplus), optionally plus a neural residual. Integrated with the same RK4
sub-stepping as the simulator, so with the true parameters it reproduces the simulator exactly.

Identifiability note (pendulum): only a=3g/(2l), c=3/(m l^2), d=3b/(m l^2) enter the dynamics, so (m,l,g)
are individually not identifiable from state/action data. `fixed` lets us hold some parameters at known values."""
import torch
import torch.nn as nn
from .base import WorldModel, rk4, DT
from .residual import Residual
from ..envs.pendulum import pendulum_deriv
from ..envs.cartpole import cartpole_deriv

PHYS = {  # env name -> (deriv fn, ordered parameter names)
    "pendulum": (pendulum_deriv, ["m", "l", "g", "b"]),
    "cartpole": (cartpole_deriv, ["M", "m", "l", "g", "b"]),
}


class PhysicsStructured(WorldModel):
    name = "physics"

    def __init__(self, env, init_params=None, use_residual=False, fixed=(), res_l2=1e-2, res_hidden=(64, 64)):
        super().__init__(env)
        self.deriv_fn, self.pnames = PHYS[env.name]
        init = dict(env.params)           # default = env's params (override to start away from truth)
        init.update(init_params or {})
        self.fixed = set(fixed)
        for k in self.pnames:
            v = init[k]
            raw = torch.tensor(float(v) if k != "b" else float(v) + 1e-3, dtype=DT).log()  # positive via exp
            if k in self.fixed:
                self.register_buffer("raw_" + k, raw)
            else:
                setattr(self, "raw_" + k, nn.Parameter(raw))
        self.use_residual, self.res_l2 = use_residual, res_l2
        if use_residual:
            self.res = Residual(self.nfeat, self.m, self.n, res_hidden)
        self.register_buffer("fscale", torch.ones(self.n, dtype=DT))

    def set_norm(self, x, xn):
        super().set_norm(x, xn); self.fscale.copy_(self.dstd / self.dt)

    # ---- parameters ----
    def get_params(self):
        out = {k: torch.exp(getattr(self, "raw_" + k)) for k in self.pnames}
        if "b" in out:
            out["b"] = (out["b"] - 1e-3).clamp_min(0.0)   # friction may be exactly 0
        return out

    def param_dict(self):
        return {k: float(v.detach()) for k, v in self.get_params().items()}

    def phys_parameters(self):
        return [getattr(self, "raw_" + k) for k in self.pnames if k not in self.fixed]

    def f(self, x, u):
        p = self.get_params()
        d = self.deriv_fn(x, u, *[p[k] for k in self.pnames])
        if self.use_residual:
            d = d + self.res(self.features(x), u / self.umax, self.fscale)
        return d

    def step(self, x, u):
        return rk4(self.f, x, u, self.dt, self.substeps)

    def loss(self, x, u, xn, idx=None):
        l = super().loss(x, u, xn, idx)
        if self.use_residual and self.res_l2 > 0:
            r = self.res(self.features(x), u / self.umax, self.fscale) * self.dt / self.dstd
            l = l + self.res_l2 * (r ** 2).mean()
        return l
