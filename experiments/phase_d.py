"""Phase D: MBRL loop for each world model + SAC on the pendulum. Saves learning curves and final models.
python -m experiments.phase_d [--seeds 5] [--iters 10]"""
import argparse, json, os, time, torch
from multiprocessing import Pool
from src.control.envs import ENVS
from src.control.agents.mbrl_agent import run_mbrl
from src.control.agents.sac_baseline import run_sac
from experiments.phase_c import make


def job(a):
    envname, name, seed, iters, sac_steps = a
    torch.set_num_threads(1); t = time.time()
    env = ENVS[envname]()
    if name == "sac":
        r = run_sac(env, seed, total_steps=sac_steps, log=lambda s: None)
        out = dict(model="sac", seed=seed, steps=r["steps"], returns=r["returns"], sec=time.time() - t)
    else:
        r = run_mbrl(envname, lambda e: make(name, e, seed), seed, n_iters=iters, log=lambda s: None)
        os.makedirs(f"results/models/{envname}", exist_ok=True)
        torch.save(r["model"].state_dict(), f"results/models/{envname}/{name}_s{seed}.pt")
        out = dict(model=name, seed=seed, steps=r["steps"], returns=r["returns"], sec=time.time() - t)
    out["env"] = envname
    print(name, seed, [round(x) for x in out["returns"]], round(out["sec"]), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="pendulum"); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10); ap.add_argument("--sac_steps", type=int, default=10_000)
    ap.add_argument("--models", nargs="+", default=["sac", "neural_hamiltonian", "legacy_latent_ode", "neural_ode", "mlp_ensemble", "physics_residual", "physics"])
    a = ap.parse_args()
    jobs = [(a.env, m, s, a.iters, a.sac_steps) for m in a.models for s in range(a.seeds)]
    with Pool(7) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"results/phase_d_{a.env}.json", "w"), indent=1)
