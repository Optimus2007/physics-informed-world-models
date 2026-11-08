"""Shared simulator machinery: batched torch RK4 + a small Sim base class.

Reuses the RK4 pattern of src/dynamics.py:rk4_step (single-state NumPy) but is
batched/torch so planners and models can roll out thousands of states at once.
Convention for all envs: state angle theta = 0 is the UPRIGHT (unstable) pose,
so "swing-up" = reach theta = 0 (mod 2*pi) and stay there. (Acrobot: theta=0 is hanging
DOWN; see acrobot.py.)
"""
import math
import torch

DTYPE = torch.float64


def angle_normalize(x):
    """Wrap angle to [-pi, pi)."""
    return ((x + math.pi) % (2 * math.pi)) - math.pi


def rk4_step_torch(f, x, u, dt, substeps=1):
    """Batched RK4. f(x,u)->dx/dt. x:(B,n) u:(B,m). Action held constant (zero-order hold)."""
    h = dt / substeps
    for _ in range(substeps):
        k1 = f(x, u)
        k2 = f(x + 0.5 * h * k1, u)
        k3 = f(x + 0.5 * h * k2, u)
        k4 = f(x + h * k3, u)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


class Sim:
    """Batched torch simulator with adjustable physics.

    params: dict of floats (physical params incl. friction `b`, torque_limit `umax`).
    """
    name = "base"
    state_dim = 0
    action_dim = 1
    dt = 0.05
    substeps = 1
    default_params = {}

    def __init__(self, **params):
        self.params = dict(self.default_params)
        self.set_params(**params)

    # --- parameter handling -------------------------------------------------
    def set_params(self, **kw):
        for k, v in kw.items():
            if k not in self.default_params:
                raise KeyError(f"unknown param {k}; valid: {list(self.default_params)}")
            self.params[k] = float(v)
        return self

    def copy(self, **kw):
        s = type(self)(**self.params)
        s.set_params(**kw)
        return s

    # --- to be implemented --------------------------------------------------
    def deriv(self, x, u, params=None):  # returns dx/dt, uses params dict (floats or tensors)
        raise NotImplementedError

    def energy(self, x):
        raise NotImplementedError

    def reward(self, x, u):
        raise NotImplementedError

    def sample_init(self, n, gen=None):
        raise NotImplementedError

    # --- common -------------------------------------------------------------
    def clip(self, u):
        um = self.params["umax"]
        return torch.clamp(u, -um, um)

    def step(self, x, u):
        """x:(B,n) u:(B,1) -> next state (B,n). Applies the true torque limit."""
        x = torch.as_tensor(x, dtype=DTYPE)
        u = self.clip(torch.as_tensor(u, dtype=DTYPE))
        return rk4_step_torch(lambda s, a: self.deriv(s, a), x, u, self.dt, self.substeps)

    def rollout(self, x0, U):
        """x0:(B,n) U:(B,H,m) -> states (B,H+1,n)."""
        xs = [x0]
        x = x0
        for t in range(U.shape[1]):
            x = self.step(x, U[:, t])
            xs.append(x)
        return torch.stack(xs, 1)
