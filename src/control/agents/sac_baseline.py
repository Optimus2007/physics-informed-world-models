"""Compact SAC for our simulators, structured after CleanRL's `sac_continuous_action.py`
(Huang et al. 2022, https://github.com/vwxyzjn/cleanrl, MIT license): tanh-Gaussian actor, twin Q networks with
target nets, automatic entropy tuning. Differences: runs on our torch sim (obs = cos, sin, omega for the pendulum),
time-limit truncation is NOT treated as terminal, and deterministic evaluation episodes are logged every
`eval_every` steps so the curve is comparable with the MPC agents' on-policy returns."""
import copy, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def obs_of(env, x):
    f = []
    for i in range(env.state_dim):
        f += [torch.cos(x[..., i:i + 1]), torch.sin(x[..., i:i + 1])] if i in env.angle_idx else [x[..., i:i + 1]]
    return torch.cat(f, -1).float()


class Actor(nn.Module):
    def __init__(self, o, a, umax):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(o, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mu, self.ls, self.umax = nn.Linear(256, a), nn.Linear(256, a), umax

    def forward(self, o, deterministic=False):
        h = self.net(o); mu, ls = self.mu(h), torch.tanh(self.ls(h)).mul(3.5).sub(1.5)  # log-std in [-5, 2]
        std = ls.exp(); x = mu if deterministic else mu + std * torch.randn_like(mu)
        a = torch.tanh(x)
        logp = (-0.5 * ((x - mu) / std) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1) - torch.log(1 - a ** 2 + 1e-6).sum(-1)
        return a * self.umax, logp


class Q(nn.Module):
    def __init__(self, o, a):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(o + a, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, o, a):
        return self.net(torch.cat([o, a], -1))[..., 0]


def run_sac(env, seed, total_steps=10_000, ep_len=200, learning_starts=1000, bs=256, gamma=0.99, tau=0.005,
            lr=1e-3, eval_every=500, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    umax = env.params["umax"]; od = obs_of(env, env.sample_start(1, g)).shape[-1]; ad = env.action_dim
    actor, q1, q2 = Actor(od, ad, umax), Q(od, ad), Q(od, ad)
    q1t, q2t = copy.deepcopy(q1), copy.deepcopy(q2)
    log_alpha = torch.zeros(1, requires_grad=True); target_ent = -float(ad)
    oa = torch.optim.Adam(actor.parameters(), lr=lr); oq = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    oal = torch.optim.Adam([log_alpha], lr=lr)
    N = total_steps; B = {k: torch.zeros(N, d) for k, d in [("o", od), ("a", ad), ("r", 1), ("on", od)]}
    steps_log, ret_log = [], []

    def evaluate():
        x = env.sample_start(1, torch.Generator().manual_seed(seed + 777))[0][None]; R = 0.0
        for _ in range(ep_len):
            with torch.no_grad():
                a, _ = actor(obs_of(env, x), True)
            R += float(env.reward(x, a.double())[0]); x = env.step(x, a.double())
        return R

    x = env.sample_start(1, g)[0][None]; t_ep = 0
    for t in range(N):
        if t % eval_every == 0:
            steps_log.append(t); ret_log.append(evaluate())
            log(f"[sac s{seed}] step {t} eval return {ret_log[-1]:.1f}")
        o = obs_of(env, x)
        if t < learning_starts:
            a = (torch.rand(1, ad) * 2 - 1) * umax
        else:
            with torch.no_grad():
                a, _ = actor(o)
        xn = env.step(x, a.double()); r = float(env.reward(x, a.double())[0])
        B["o"][t], B["a"][t], B["r"][t], B["on"][t] = o[0], a[0], r, obs_of(env, xn)[0]
        x = xn; t_ep += 1
        if t_ep >= ep_len:
            x = env.sample_start(1, g)[0][None]; t_ep = 0
        if t >= learning_starts:
            i = torch.randint(0, t + 1, (bs,)); o_, a_, r_, on_ = (B[k][i] for k in ("o", "a", "r", "on"))
            with torch.no_grad():
                an, lp = actor(on_); alpha = log_alpha.exp()
                tq = torch.min(q1t(on_, an), q2t(on_, an)) - alpha * lp
                y = r_[:, 0] + gamma * tq
            lq = F.mse_loss(q1(o_, a_), y) + F.mse_loss(q2(o_, a_), y)
            oq.zero_grad(); lq.backward(); oq.step()
            ap, lp = actor(o_); la = (alpha * lp - torch.min(q1(o_, ap), q2(o_, ap))).mean()
            oa.zero_grad(); la.backward(); oa.step()
            lal = (-log_alpha.exp() * (lp.detach() + target_ent)).mean()
            oal.zero_grad(); lal.backward(); oal.step()
            for n_, t_ in ((q1, q1t), (q2, q2t)):
                for p, pt in zip(n_.parameters(), t_.parameters()):
                    pt.data.mul_(1 - tau).add_(tau * p.data)
    steps_log.append(N); ret_log.append(evaluate())
    return dict(steps=steps_log, returns=ret_log)
