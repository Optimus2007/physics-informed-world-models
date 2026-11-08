"""Phase C: train every world model on the same random-action dataset; log 1-step/multi-step error + energy drift.
Usage: python -m experiments.phase_c [--env pendulum] [--seeds 5] [--sizes 8 20 100] [--steps 1500] [--out results/phase_c_pendulum.json]
Dataset size is in EPISODES of 50 transitions (8 -> 400, 20 -> 1000, 100 -> 5000 transitions)."""
import argparse, json, time, torch
from multiprocessing import Pool
from src.control.envs import ENVS
from src.control.eval.data import collect_random, fit
from src.control.eval.rollouts import eval_data, horizon_error, energy_drift
from src.control.models.mlp_ensemble import MLPEnsemble
from src.control.models.neural_ode import NeuralODE
from src.control.models.physics_structured import PhysicsStructured
from src.control.models.neural_hamiltonian import NeuralHamiltonian
from src.control.models.legacy_latent_ode import LegacyLatentODE


def make(name, env, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    # physics models start from WRONG parameters: truth x U(0.6, 1.4)
    wrong = {k: v * (0.6 + 0.8 * float(torch.rand(1, generator=g))) for k, v in env.params.items() if k in "Mmlg"}
    wrong["b"] = 0.0     # friction is unknown to the agent: never initialise it from the env's true value
    return {"mlp_ensemble": lambda: MLPEnsemble(env), "neural_ode": lambda: NeuralODE(env),
            "physics": lambda: PhysicsStructured(env, init_params=wrong),
            "physics_residual": lambda: PhysicsStructured(env, init_params=wrong, use_residual=True),
            # Phase F: friction term absent from the hypothesis class (b frozen at 0)
            "physics_nofric": lambda: PhysicsStructured(env, init_params=wrong, fixed=("b",)),
            "physics_nofric_residual": lambda: PhysicsStructured(env, init_params=wrong, fixed=("b",), use_residual=True),
            "neural_hamiltonian": lambda: NeuralHamiltonian(env),
            "legacy_latent_ode": lambda: LegacyLatentODE(env)}[name]()


def job(a):
    envname, name, size, seed, steps = a
    torch.set_num_threads(1); torch.manual_seed(seed)
    env = ENVS[envname]()
    data = collect_random(env, size, 50, seed=seed)
    ev = eval_data(env)
    m = make(name, env, seed)
    t = time.time(); r = fit(m, data, steps=steps, seed=seed)
    out = dict(env=envname, model=name, size=size, n_transitions=len(data), seed=seed, val=r["best_val"],
               horizon=horizon_error(m, env, *ev), energy=energy_drift(m, env), sec=time.time() - t,
               params=m.param_dict() if hasattr(m, "param_dict") else None)
    print(name, size, seed, round(out["sec"]), "val", f"{out['val']:.2e}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="pendulum"); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 20, 100]); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--models", nargs="+", default=["mlp_ensemble", "neural_ode", "physics", "physics_residual", "neural_hamiltonian", "legacy_latent_ode"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    jobs = [(a.env, m, s, sd, a.steps) for s in a.sizes for sd in range(a.seeds) for m in a.models]
    jobs.sort(key=lambda j: j[1] != "neural_hamiltonian")      # slowest first
    with Pool(7) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(a.out or f"results/phase_c_{a.env}.json", "w"), indent=1)
