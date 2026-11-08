"""Cross-check own MPPI vs pytorch_mppi (UM-ARM-Lab, MIT) on the TRUE pendulum. Writes results/crosscheck_mppi.json."""
import json, math, torch
from pytorch_mppi import MPPI as LibMPPI
from src.control.envs import Pendulum
from src.control.planners.mppi import MPPI, run_episode

env = Pendulum(); x0 = torch.tensor([math.pi, 0.0], dtype=torch.float64)
res = {"own": [], "lib": []}
for seed in range(5):
    own = MPPI(env.step, env.reward, horizon=40, n_samples=500, lam=0.5, sigma=1.0, umax=2.0, seed=seed)
    X, U, R = run_episode(env, own, x0, 200); res["own"].append(float(R.sum()))
    torch.manual_seed(seed)
    ctl = LibMPPI(lambda s, a: env.step(s.double(), a.double()), lambda s, a: -env.reward(s.double(), a.double()),
                  2, torch.tensor([[1.0]], dtype=torch.double), 500, 40, lambda_=0.5,
                  u_min=torch.tensor([-2.0], dtype=torch.double), u_max=torch.tensor([2.0], dtype=torch.double),
                  device="cpu", noise_mu=torch.zeros(1, dtype=torch.double))
    x, tot = x0.clone()[None], 0.0
    for _ in range(200):
        u = ctl.command(x[0]).double()[None]
        tot += float(env.reward(x, u)[0]); x = env.step(x, u)
    res["lib"].append(tot)
res["note"] = "returns over 200 steps from hanging start; lower magnitude = better"
print(res); json.dump(res, open("results/crosscheck_mppi.json", "w"), indent=1)
