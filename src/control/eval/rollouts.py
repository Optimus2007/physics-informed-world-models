"""Prediction-error and energy-drift evaluation of a world model against the true simulator."""
import math, torch
from .data import DT
from ..envs.base import angle_normalize


def state_err(env, a, b):
    """Per-dim difference with angle coordinates wrapped; a,b:(...,n)."""
    d = a - b
    for i in env.angle_idx:
        d[..., i] = angle_normalize(d[..., i])
    return d


def eval_data(env, n=200, H=100, seed=1234, wide=True):
    """Fixed evaluation set: wide initial states + random actions (seed disjoint from training seeds)."""
    g = torch.Generator().manual_seed(seed)
    x0 = env.sample_init_wide(n, g) if wide else env.sample_init(n, g)
    U = (torch.rand(n, H, env.action_dim, generator=g, dtype=DT) * 2 - 1) * env.params["umax"]
    return x0, U, env.rollout(x0, U)


@torch.no_grad()
def horizon_error(model, env, x0, U, true_traj, horizons=(1, 10, 25, 50, 100)):
    """RMSE (all state dims, angles wrapped) at each horizon, normalised per-dim by the std of the true traj.
    Returns dict {H: rmse}; also the median over initial states (robust to chaotic outliers)."""
    pred = model.rollout(x0, U)
    sd = true_traj.reshape(-1, true_traj.shape[-1]).std(0)
    out = {}
    for h in horizons:
        if h >= pred.shape[1]:
            continue
        e = state_err(env, pred[:, h], true_traj[:, h]) / sd
        per = e.pow(2).mean(-1).sqrt()
        out[h] = dict(mean=float(per.mean()), median=float(per.median()))
    return out


@torch.no_grad()
def energy_drift(model, env, n=50, T=200, seed=99, wide=False):
    """Zero control, env frictionless: |E_true(x_t) - E_true(x_0)| / (|E_true(x0)| + 1), true energy evaluated on
    MODEL states (a physical-consistency metric that does not need the model to have its own energy). Median over
    initial conditions, returned at several times."""
    g = torch.Generator().manual_seed(seed)
    x0 = env.sample_init(n, g) if not wide else env.sample_init_wide(n, g)
    if env.name == "cartpole":     # start from moderate swings rather than near-hanging only
        x0[:, 1] = (torch.rand(n, generator=g, dtype=DT) * 2 - 1) * math.pi
    U = torch.zeros(n, T, env.action_dim, dtype=DT)
    pred = model.rollout(x0, U)
    E0 = env.energy(x0)
    ts = [t for t in (10, 50, 100, 200) if t <= T]
    return {t: float(((env.energy(pred[:, t]) - E0).abs() / (E0.abs() + 1)).median()) for t in ts}
