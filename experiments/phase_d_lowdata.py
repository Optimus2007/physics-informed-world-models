"""H1 low-data sweep: fit each model on only k random episodes (50 transitions each), then plan ONE episode.
Return of that episode measures how good a model fitted on k*50 transitions is for control."""
import argparse, json, time, torch
from multiprocessing import Pool
from src.control.envs import Pendulum
from src.control.agents.mbrl_agent import run_mbrl
from experiments.phase_c import make


def job(a):
    name, k, seed = a
    torch.set_num_threads(1); t = time.time()
    r = run_mbrl("pendulum", lambda e: make(name, e, seed), seed, n_init_eps=k, n_iters=1, log=lambda s: None)
    out = dict(model=name, n_init_eps=k, n_transitions=k * 50, seed=seed, ret=r["returns"][0], sec=time.time() - t)
    print(name, k, seed, round(out["ret"]), round(out["sec"]), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4]); a = ap.parse_args()
    models = ["neural_hamiltonian", "legacy_latent_ode", "neural_ode", "mlp_ensemble", "physics_residual", "physics"]
    jobs = [(m, k, s) for m in models for k in a.ks for s in range(a.seeds)]
    with Pool(7) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open("results/phase_d_lowdata_pendulum.json", "w"), indent=1)
