"""Phase E: changed physics. Models trained at DEFAULT physics (from phase_d) are evaluated in shifted envs,
(a) unchanged and (b) after adapting with 50 new transitions. Also: transitions-needed curves.
python -m experiments.phase_e [--seeds 5]"""
import argparse, json, math, time, torch
from multiprocessing import Pool
from src.control.envs import Pendulum
from src.control.agents.mbrl_agent import act_in
from src.control.eval.data import fit
from src.control.eval.rollouts import eval_data, horizon_error
from src.control.adapt.sysid import adapt_physics, adapt_blackbox, collect_adaptation_data
from src.control.planners.mppi import MPPI, run_episode
from experiments.phase_c import make

SHIFTS = {"g x0.5": dict(g=5.0), "g x1.5": dict(g=15.0), "g x2.0": dict(g=20.0),
          "l x0.7": dict(l=0.7), "l x1.5": dict(l=1.5), "m x0.5": dict(m=0.5), "m x2.0": dict(m=2.0)}
MODELS = ["physics", "physics_residual", "neural_hamiltonian", "neural_ode", "mlp_ensemble", "legacy_latent_ode"]
LRS = [1e-4, 3e-4, 1e-3, 3e-3]
NS = [25, 50, 100, 200, 500, 1000]


def load(name, seed):
    env = Pendulum(); m = make(name, env, seed)
    m.load_state_dict(torch.load(f"results/models/pendulum/{name}_s{seed}.pt")); m.eval()
    return m


def h25(model, env, ev):
    return horizon_error(model, env, *ev, horizons=(1, 25))


def episodes(env, model, seed, n=2):
    g = torch.Generator().manual_seed(500 + seed)
    rets = []
    for i in range(n):
        x0 = env.sample_start(1, g)[0]
        rets.append(float(act_in(env, model, x0, 200, seed * 100 + i)[2].sum()))
    return sum(rets) / n


def tune_lr(model, seed):
    """Pick the black-box fine-tuning lr on a SEPARATE tuning shift (g x0.75, l x1.2), by H=25 error."""
    tenv = Pendulum(g=7.5, l=1.2); ev = eval_data(tenv, n=60, H=30, seed=4242)
    data = collect_adaptation_data(tenv, 50, 5, seed=900 + seed)
    best = min(LRS, key=lambda lr: h25(adapt_blackbox(model, data, lr=lr), tenv, ev)[25]["median"])
    return best


def job(a):
    name, seed = a
    torch.set_num_threads(1); torch.manual_seed(seed); t0 = time.time()
    base = load(name, seed); lr = tune_lr(base, seed) if name != "physics" and name != "physics_residual" else None
    rows = []
    for sname, sh in SHIFTS.items():
        env = Pendulum(**sh); ev = eval_data(env, n=100, H=30, seed=777)
        data = collect_adaptation_data(env, 50, 5, seed=seed)
        row = dict(model=name, seed=seed, shift=sname, true=sh)
        row["err_before"] = h25(base, env, ev)
        row["ret_noadapt"] = episodes(env, base, seed)
        if name.startswith("physics"):
            ad = adapt_physics(base, data, steps=400, lr=3e-2)
            p = ad.param_dict(); row["params"] = p
            row["a"] = 1.5 * p["g"] / p["l"]; row["c"] = 3 / (p["m"] * p["l"] ** 2)
            row["a_true"] = 1.5 * env.params["g"] / env.params["l"]; row["c_true"] = 3 / (env.params["m"] * env.params["l"] ** 2)
            if name == "physics":    # oracle-knowledge variant: adapt only the parameter that really changed
                knownk = list(sh)[0]
                ad2 = adapt_physics(base, data, steps=400, lr=3e-2, which=[knownk])
                row["ret_adapt_known"] = episodes(env, ad2, seed); row["params_known"] = ad2.param_dict()
        else:
            ad = adapt_blackbox(base, data, lr=lr); row["lr"] = lr
        row["err_after"] = h25(ad, env, ev); row["ret_adapt"] = episodes(env, ad, seed)
        pl = MPPI(env.step, env.reward, horizon=40, n_samples=400, lam=0.5, sigma=1.0, umax=2.0, seed=seed)
        g = torch.Generator().manual_seed(500 + seed)
        row["ret_oracle"] = float(run_episode(env, pl, env.sample_start(1, g)[0], 200)[2].sum())
        # transitions needed (prediction error at H=25 vs number of new transitions)
        curve = {}
        for n in NS:
            d = collect_adaptation_data(env, n, max(1, n // 10), seed=seed)
            if name.startswith("physics"):
                m_ = adapt_physics(base, d, steps=400, lr=3e-2)
            else:
                m_ = adapt_blackbox(base, d, lr=lr)
            curve[n] = h25(m_, env, ev)[25]["median"]
        row["curve"] = curve
        rows.append(row)
        print(name, seed, sname, "noadapt", round(row["ret_noadapt"]), "adapt", round(row["ret_adapt"]), "oracle", round(row["ret_oracle"]), flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    jobs = [(m, s) for m in MODELS for s in range(a.seeds)]
    with Pool(7) as p:
        res = [r for rows in p.map(job, jobs, chunksize=1) for r in rows]
    json.dump(res, open("results/phase_e_pendulum.json", "w"), indent=1)
