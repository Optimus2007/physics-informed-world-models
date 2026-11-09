"""Side-by-side GIF at shifted gravity (g x1.5): MLP ensemble unadapted | MLP ensemble adapted (50 transitions) |
physics model adapted (50 transitions). Uses models saved by experiments/phase_d.py (seed 0). One 200-step (10 s) episode."""
import math, torch, numpy as np, imageio.v2 as imageio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.control.envs import Pendulum
from src.control.agents.mbrl_agent import act_in
from src.control.adapt.sysid import adapt_physics, adapt_blackbox, collect_adaptation_data
from experiments.phase_e import load

SEED = 0; env = Pendulum(g=15.0)
data = collect_adaptation_data(env, 50, 5, seed=SEED)
mlp = load("mlp_ensemble", SEED); phys = load("physics", SEED)
agents = {"MLP ensemble, no adaptation": mlp, "MLP ensemble, adapted (50 trans.)": adapt_blackbox(mlp, data, lr=1e-3),
          "Physics model, adapted (50 trans.)": adapt_physics(phys, data)}
x0 = torch.tensor([math.pi + 0.1, 0.0], dtype=torch.float64)
runs = {k: act_in(env, m, x0, 200, 0) for k, m in agents.items()}
frames = []
for t in range(0, 200, 2):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.4))
    for ax, (k, (X, U, R)) in zip(axs, runs.items()):
        th = float(X[t, 0]); ax.plot([0, math.sin(th)], [0, math.cos(th)], "-o", lw=4, color="#0072B2")
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{k}\nt={t*0.05:.1f}s  return={float(R[:t].sum()):.0f}", fontsize=8)
    fig.suptitle("Pendulum swing-up with gravity x1.5 (never seen in training)", fontsize=10)
    fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
imageio.mimsave("figures/demo_shifted_gravity.gif", frames, fps=10, loop=0)
print({k: round(float(v[2].sum())) for k, v in runs.items()})
