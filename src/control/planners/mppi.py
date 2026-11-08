"""Batched MPPI (Williams et al., 2016/2017) and CEM, model-agnostic.

Interface: step_fn(x:(N,n), u:(N,m)) -> (N,n);  reward_fn(x:(N,n), u:(N,m)) -> (N,)
Our own short implementation so every line can be explained in the report; cross-checked against the
`pytorch_mppi` package in experiments/crosscheck_mppi.py.
"""
import torch

DT = torch.float64


class MPPI:
    def __init__(self, step_fn, reward_fn, action_dim=1, horizon=30, n_samples=500,
                 lam=1.0, sigma=1.0, umax=2.0, gamma=1.0, seed=0):
        self.step_fn, self.reward_fn = step_fn, reward_fn
        self.m, self.H, self.N = action_dim, horizon, n_samples
        self.lam, self.sigma, self.umax, self.gamma = lam, sigma, umax, gamma
        self.gen = torch.Generator().manual_seed(seed)
        self.reset()

    def reset(self):
        self.U = torch.zeros(self.H, self.m, dtype=DT)  # nominal action sequence

    @torch.no_grad()
    def plan(self, x0):
        """x0:(n,) -> first action (m,). Updates nominal sequence (receding horizon)."""
        N, H, m = self.N, self.H, self.m
        eps = torch.randn(N, H, m, generator=self.gen, dtype=DT) * self.sigma
        eps[0] = 0.0                                          # keep the unperturbed nominal as a candidate
        A = torch.clamp(self.U[None] + eps, -self.umax, self.umax)   # (N,H,m) candidate sequences
        eps = A - self.U[None]                                # effective (post-clip) perturbation
        x = x0.to(DT)[None].expand(N, -1).clone()
        R = torch.zeros(N, dtype=DT)
        for t in range(H):                                    # roll all N sequences in parallel
            R = R + (self.gamma ** t) * self.reward_fn(x, A[:, t])
            x = self.step_fn(x, A[:, t])
        # information-theoretic weights: w_i ∝ exp(R_i / lam); subtract max for numerical stability
        w = torch.softmax((R - R.max()) / self.lam, dim=0)
        self.U = self.U + (w[:, None, None] * eps).sum(0)     # weighted average of perturbations
        self.U = torch.clamp(self.U, -self.umax, self.umax)
        u0 = self.U[0].clone()
        self.U = torch.cat([self.U[1:], torch.zeros(1, m, dtype=DT)], 0)   # shift for next step
        self.last_return = R
        return u0


class CEM:
    """Cross-entropy method: refit Gaussian to top-k elite sequences, iterate."""
    def __init__(self, step_fn, reward_fn, action_dim=1, horizon=30, n_samples=500, n_elite=50,
                 iters=4, sigma=1.0, umax=2.0, seed=0):
        self.step_fn, self.reward_fn = step_fn, reward_fn
        self.m, self.H, self.N, self.K, self.iters = action_dim, horizon, n_samples, n_elite, iters
        self.sigma0, self.umax = sigma, umax
        self.gen = torch.Generator().manual_seed(seed)
        self.reset()

    def reset(self):
        self.mu = torch.zeros(self.H, self.m, dtype=DT)

    @torch.no_grad()
    def plan(self, x0):
        mu, sd = self.mu.clone(), torch.full_like(self.mu, self.sigma0)
        for _ in range(self.iters):
            A = torch.clamp(mu[None] + sd[None] * torch.randn(self.N, self.H, self.m, generator=self.gen, dtype=DT),
                            -self.umax, self.umax)
            x = x0.to(DT)[None].expand(self.N, -1).clone()
            R = torch.zeros(self.N, dtype=DT)
            for t in range(self.H):
                R = R + self.reward_fn(x, A[:, t]); x = self.step_fn(x, A[:, t])
            el = A[R.topk(self.K).indices]
            mu, sd = el.mean(0), el.std(0) + 1e-3
        u0 = mu[0].clone()
        self.mu = torch.cat([mu[1:], torch.zeros(1, self.m, dtype=DT)], 0)
        return u0


def run_episode(sim, planner, x0, steps, true_sim=None):
    """Closed-loop receding-horizon control on `true_sim` (default: sim). Returns states, actions, rewards."""
    env = true_sim or sim
    planner.reset()
    x = x0.clone().reshape(1, -1).to(DT)
    X, U, Rw = [x[0].clone()], [], []
    for _ in range(steps):
        u = planner.plan(x[0])
        U.append(u.clone())
        Rw.append(env.reward(x, u[None])[0])
        x = env.step(x, u[None])
        X.append(x[0].clone())
    return torch.stack(X), torch.stack(U), torch.stack(Rw)
