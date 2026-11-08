"""Common world-model interface. All models are float64 (matches the simulators).

step(x:(B,n), u:(B,m)) -> (B,n)           next state after one control interval dt
rollout(x0:(B,n), U:(B,H,m)) -> (B,H+1,n)
loss(x,u,xn,idx) -> scalar                 one-step loss, normalised by per-dim std of the state increment
Angles live in the state UNWRAPPED; networks see (cos, sin) of angle coordinates, so they are periodic.
"""
import torch
import torch.nn as nn

DT = torch.float64


def mlp(inp, out, hidden=(128, 128), act=nn.SiLU, last_scale=None):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out))
    net = nn.Sequential(*layers).double()
    if last_scale is not None:                       # small init of the output layer (e.g. residuals ~ 0)
        with torch.no_grad():
            net[-1].weight.mul_(last_scale); net[-1].bias.zero_()
    return net


class WorldModel(nn.Module):
    name = "base"

    def __init__(self, env):
        super().__init__()
        self.n, self.m = env.state_dim, env.action_dim
        self.angle_idx = tuple(env.angle_idx)
        self.dt, self.substeps, self.umax = env.dt, env.substeps, env.params["umax"]
        self.register_buffer("dstd", torch.ones(self.n, dtype=DT))
        self.register_buffer("dmean", torch.zeros(self.n, dtype=DT))

    # ---- shared helpers ----
    def features(self, x):
        """Replace each angle by (cos, sin). (B,n) -> (B, n + len(angles))."""
        cols = []
        for i in range(self.n):
            if i in self.angle_idx:
                cols += [torch.cos(x[..., i:i + 1]), torch.sin(x[..., i:i + 1])]
            else:
                cols.append(x[..., i:i + 1])
        return torch.cat(cols, -1)

    @property
    def nfeat(self):
        return self.n + len(self.angle_idx)

    def set_norm(self, x, xn):
        d = xn - x
        self.dmean.copy_(d.mean(0)); self.dstd.copy_(d.std(0).clamp_min(1e-3))

    def step(self, x, u):
        raise NotImplementedError

    def loss(self, x, u, xn, idx=None):
        return (((self.step(x, u) - xn) / self.dstd) ** 2).mean()

    @torch.no_grad()
    def rollout(self, x0, U):
        xs, x = [x0], x0
        for t in range(U.shape[1]):
            x = self.step(x, U[:, t]); xs.append(x)
        return torch.stack(xs, 1)


def rk4(f, x, u, dt, substeps):
    h = dt / substeps
    for _ in range(substeps):
        k1 = f(x, u); k2 = f(x + 0.5 * h * k1, u); k3 = f(x + 0.5 * h * k2, u); k4 = f(x + h * k3, u)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
