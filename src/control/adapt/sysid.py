"""System identification / adaptation from a handful of new transitions.

Structured model: freeze everything except the physical parameters (raw log-params) and run Adam on the
one-step loss over the new transitions. This is a small nonlinear least-squares problem
   min_theta  sum_i || x_{i+1} - RK4(x_i, u_i; theta) ||^2 / sigma^2
with 1-5 unknowns, so ~50 transitions suffice. Black-box models are fine-tuned on the same transitions
(all weights), early-stopped on a held-out episode, lr chosen from a small grid on a *separate* tuning shift."""
import copy, torch
from ..eval.data import fit, Data, collect_random


def collect_adaptation_data(env, n_transitions=50, n_episodes=5, seed=0):
    """n_episodes short random-action episodes from wide initial states in the (shifted) env."""
    return collect_random(env, n_episodes, n_transitions // n_episodes, seed=10_000 + seed, wide=True)


def adapt_physics(model, data, steps=400, lr=3e-2, which=None):
    """Re-estimate physical parameters only. which: iterable of parameter names to adapt (default: all non-fixed)."""
    m = copy.deepcopy(model)
    names = which or [k for k in m.pnames if k not in m.fixed]
    for p in m.parameters():
        p.requires_grad_(False)
    ps = [getattr(m, "raw_" + k) for k in names]
    for p in ps:
        p.requires_grad_(True)
    fit(m, data, steps=steps, lr=lr, val_frac=0.0, params=ps, renorm=False, patience=10 ** 9, eval_every=steps)
    return m


def adapt_blackbox(model, data, steps=300, lr=1e-3):
    m = copy.deepcopy(model)
    fit(m, data, steps=steps, lr=lr, val_frac=0.2, renorm=False, patience=3, eval_every=25)
    return m
