"""Phase F: where the physics prior fails. 'Real' pendulum has viscous friction b and a HIDDEN actuator saturation
usat (< the planner's command range umax). Every model learns from scratch via the MBRL loop in this env.
physics_nofric: friction term absent (b frozen at 0), saturation never representable;
physics_nofric_residual: + neural residual; physics: friction learnable (but still no saturation)."""
import argparse, json, time, torch
from multiprocessing import Pool
from src.control.envs import Pendulum
from src.control.agents.mbrl_agent import run_mbrl
from src.control.eval.rollouts import eval_data, horizon_error
from src.control.planners.mppi import MPPI, run_episode
from experiments.phase_c import make

REAL = dict(b=0.15, usat=1.5)
MODELS = ["neural_hamiltonian", "neural_ode", "mlp_ensemble", "physics_nofric_residual", "physics_nofric", "physics", "oracle"]


def job(a):
    name, seed, iters = a
    torch.set_num_threads(1); t = time.time(); env = Pendulum(**REAL)
    if name == "oracle":     # true simulator as the model, in the real env
        g = torch.Generator().manual_seed(seed); rets = []
        for i in range(iters):
            pl = MPPI(env.step, env.reward, horizon=40, n_samples=400, lam=0.5, sigma=1.0, umax=2.0, seed=seed * 1000 + i)
            rets.append(float(run_episode(env, pl, env.sample_start(1, g)[0], 200)[2].sum()))
        out = dict(model=name, seed=seed, steps=[0] * iters, returns=rets)
    else:
        r = run_mbrl("pendulum", lambda e: make(name, e, seed), seed, n_iters=iters, true_env=env, log=lambda s: None)
        m = r["model"]; ev = eval_data(env, n=100, H=30, seed=777)
        out = dict(model=name, seed=seed, steps=r["steps"], returns=r["returns"], err=horizon_error(m, env, *ev, horizons=(1, 25)),
                   params=m.param_dict() if hasattr(m, "param_dict") else None)
    out["sec"] = time.time() - t
    print(name, seed, [round(x) for x in out["returns"]], round(out["sec"]), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    jobs = [(m, s, a.iters) for m in MODELS for s in range(a.seeds)]
    with Pool(7) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open("results/phase_f_pendulum.json", "w"), indent=1)
