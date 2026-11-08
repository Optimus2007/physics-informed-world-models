"""GATE B: MPPI/CEM with the TRUE simulator as the model. Writes results/gate_b.json."""
import json, math, sys, time, torch
from src.control.envs import Pendulum, CartPole
from src.control.planners.mppi import MPPI, CEM, run_episode


def upright_stats(env, X, k=50):
    th = (X[-k:, 1] if env.name == "cartpole" else X[-k:, 0])
    th = (th + math.pi) % (2 * math.pi) - math.pi
    return float(th.abs().mean()), float(th.abs().max())


def trial(env_cls, planner_cls, cfg, steps, seed):
    env = env_cls(); g = torch.Generator().manual_seed(seed)
    x0 = env.sample_init(1, g)[0]
    if env.name == "pendulum":
        x0 = torch.tensor([math.pi, 0.0], dtype=torch.float64)   # hanging, at rest: hardest start
    pl = planner_cls(env.step, env.reward, horizon=cfg["H"], n_samples=cfg["N"], sigma=cfg["sigma"],
                     umax=env.params["umax"], seed=seed, **cfg.get("extra", {}))
    t = time.time(); X, U, R = run_episode(env, pl, x0, steps)
    mean_err, max_err = upright_stats(env, X)
    return dict(seed=seed, ret=float(R.sum()), mean_abs_angle_last50=mean_err, max_abs_angle_last50=max_err,
                solved=bool(max_err < 0.3), sec=time.time() - t)


if __name__ == "__main__":
    out = {}
    cfgs = {
        "pendulum_mppi": (Pendulum, MPPI, dict(H=40, N=500, sigma=1.0, extra=dict(lam=0.5)), 200),
        "pendulum_cem": (Pendulum, CEM, dict(H=40, N=500, sigma=1.0), 200),
        "cartpole_mppi": (CartPole, MPPI, dict(H=50, N=1000, sigma=4.0, extra=dict(lam=0.5)), 300),
    }
    sel = sys.argv[1:] or list(cfgs)
    for k in sel:
        ec, pc, cfg, steps = cfgs[k]
        out[k] = dict(cfg={a: b for a, b in cfg.items()}, steps=steps, trials=[trial(ec, pc, cfg, steps, s) for s in range(5)])
        print(k, [(round(t["ret"], 1), t["solved"], round(t["sec"], 1)) for t in out[k]["trials"]], flush=True)
    json.dump(out, open("results/gate_b_%s.json" % ("_".join(sel) if len(sel) < 3 else "all"), "w"), indent=1)
