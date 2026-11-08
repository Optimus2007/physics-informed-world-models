"""Model-based RL loop: random data -> fit model -> plan with MPPI in the model -> execute in the TRUE env ->
append transitions -> refit (warm start) -> repeat. The reward function is known to the agent (standard in
PETS/MPC); only the dynamics are learned."""
import copy, time, torch
from ..envs import ENVS
from ..eval.data import Data, collect_random, fit
from ..planners.mppi import MPPI, run_episode

DT = torch.float64
PLAN = dict(pendulum=dict(H=40, N=400, lam=0.5, sigma=1.0),
            cartpole=dict(H=50, N=600, lam=0.5, sigma=4.0))


def make_planner(model, env, cfg, seed):
    return MPPI(model.step, env.reward, horizon=cfg["H"], n_samples=cfg["N"], lam=cfg["lam"], sigma=cfg["sigma"],
                umax=env.params["umax"], seed=seed)


def act_in(true_env, model, x0, steps, seed, cfg=None):
    """Closed-loop MPC episode of `model` in `true_env`. Returns X, U, R tensors."""
    cfg = cfg or PLAN[true_env.name]
    return run_episode(true_env, make_planner(model, true_env, cfg, seed), x0, steps)


def run_mbrl(env_name, make_model, seed, n_init_eps=10, n_iters=10, ep_len=200, init_steps=1500, iter_steps=300,
             true_env=None, log=print):
    """Returns dict(steps=[...], returns=[...], solved=[...]) where entry i = performance of the model fitted on
    `steps[i]` environment transitions (incl. the random ones), measured by the next on-policy episode."""
    torch.manual_seed(seed)
    env = true_env or ENVS[env_name]()
    g = torch.Generator().manual_seed(seed)
    data = collect_random(env, n_init_eps, 50, seed=seed)
    n_steps = len(data)
    model = make_model(env)
    fit(model, data, steps=init_steps, seed=seed)
    out = dict(steps=[], returns=[], seconds=[])
    for it in range(n_iters):
        t0 = time.time()
        x0 = env.sample_start(1, g)[0]
        X, U, R = act_in(env, model, x0, ep_len, seed * 1000 + it)
        out["steps"].append(n_steps); out["returns"].append(float(R.sum()))
        data.add(X[:-1], U, X[1:], 10_000 + it); n_steps += ep_len
        fit(model, data, steps=iter_steps, seed=seed + it, renorm=False)
        out["seconds"].append(time.time() - t0)
        log(f"[{model.name} s{seed}] it {it} steps {out['steps'][-1]} return {out['returns'][-1]:.1f} ({out['seconds'][-1]:.0f}s)")
    out["model"] = model
    return out
