"""Regenerate the two figures used on the portfolio site, from this repo's code and results.

    python analysis/make_site_figures.py            # Part I retrain (CPU, a few minutes) + Part II from results/
    python analysis/make_site_figures.py --skip-orbit   # Part II only (reads results/phase_d_pendulum.json)

Outputs
    results/site_figures.json          numbers the site draws (no hand-edited values)
    figures/site_orbit_error.png       Part I: position error vs rollout step, MLP / PINN / PIWM
    figures/site_learning_curves.png   Part II: swing-up return vs environment steps, model-based vs SAC

Part I re-runs the three training scripts exactly as written (same data, architectures, epochs, learning rates
and losses), with a fixed seed. Note that src/train_naive.py fits the one-step MLP on the full trajectory,
the PINN fits the first 2,000 and the PIWM only the first 400 of 10,000 steps; the JSON records this.
"""
import argparse, collections, json, os, sys, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
os.chdir(ROOT)

SEED, DT, N_STEPS, TRAIN_CUTOFF, EVERY = 0, 0.01, 10000, 400, 20


def orbit_part():
    import torch, torch.nn as nn, torch.optim as optim
    torch.set_num_threads(2)
    from data.generate_orbit import simulate_orbit
    from train_naive import NaiveMLP
    from pinns import OrbitalPINN
    from models import PhysicsInformedWorldModel

    truth = simulate_orbit(np.array([1.0, 0.0, 0.0, 1.2]), DT, N_STEPS)
    t = torch.tensor(np.arange(0, N_STEPS * DT, DT)[:N_STEPS], dtype=torch.float32)
    x = torch.tensor(truth, dtype=torch.float32)
    preds, secs = {}, {}

    # naive MLP: as src/train_naive.py (one-step pairs over the whole trajectory, 1500 epochs, Adam 1e-3)
    torch.manual_seed(SEED); t0 = time.time()
    m = NaiveMLP(); opt = optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(1500):
        opt.zero_grad(); loss = nn.functional.mse_loss(m(x[:-1]), x[1:]); loss.backward(); opt.step()
    with torch.no_grad():
        out = torch.zeros_like(x); out[0] = x[0]; cur = x[:1]
        for i in range(1, N_STEPS):
            cur = m(cur); out[i] = cur[0]
    preds["mlp"] = out.numpy(); secs["mlp"] = time.time() - t0

    # PINN: as src/train_pinn.py (first 2,000 of 10,000 steps of data, 2000 collocation points, 3000 epochs)
    torch.manual_seed(SEED); t0 = time.time()
    tt = t.reshape(-1, 1); cut = int(N_STEPS * 0.2)
    p = OrbitalPINN(hidden_dim=64); opt = optim.Adam(p.parameters(), lr=1e-3); tmax = tt[-1].item()
    for _ in range(3000):
        opt.zero_grad()
        ld = nn.functional.mse_loss(p(tt[:cut]), x[:cut])
        tc = torch.rand((2000, 1), requires_grad=True) * tmax
        (ld + 0.1 * p.physics_loss(tc)).backward(); opt.step()
    with torch.no_grad():
        preds["pinn"] = p(tt).numpy()
    secs["pinn"] = time.time() - t0

    # PIWM: as src/train_world_model.py (first 400 steps, dopri5, energy-variance loss 0.1, clip 1.0, 1000 epochs)
    torch.manual_seed(SEED); t0 = time.time()
    w = PhysicsInformedWorldModel(state_dim=4, latent_dim=4, hidden_dim=64); opt = optim.Adam(w.parameters(), lr=1e-3)
    xt = x[:TRAIN_CUTOFF].unsqueeze(1); x0 = xt[0]
    for ep in range(1000):
        opt.zero_grad(); xp = w(x0, t[:TRAIN_CUTOFF])
        loss = nn.functional.mse_loss(xp, xt) + 0.1 * torch.var(w.compute_energy(xp))
        loss.backward(); torch.nn.utils.clip_grad_norm_(w.parameters(), 1.0); opt.step()
        if ep % 100 == 0: print(f"  piwm epoch {ep} loss {loss.item():.5f}", flush=True)
    w.eval()
    with torch.no_grad():
        preds["piwm"] = w(x0, t).squeeze(1).numpy()
    secs["piwm"] = time.time() - t0

    idx = np.arange(0, N_STEPS, EVERY)
    curve = truth[:, :2]  # the true orbit traced many times over; its path is the reference shape
    def shape_err(pts):
        return np.array([np.min(np.linalg.norm(curve - q, axis=1)) for q in pts])
    err = {k: np.linalg.norm(v[:, :2] - truth[:, :2], axis=1) for k, v in preds.items()}
    return {
        "system": "two-body orbit, GM=1, x0=(1,0,0,1.2), RK4 dt=0.01, 10,000 steps",
        "seed": SEED, "train_steps": {"mlp": "all 10,000 (one-step pairs, as src/train_naive.py)",
                                      "pinn": "first 2,000 of 10,000 (as src/train_pinn.py)",
                                      "piwm": "first 400 (as src/train_world_model.py)"},
        "metric": "Euclidean position error |(x,y)_pred - (x,y)_true| along a free rollout from the initial state",
        "steps": idx.tolist(),
        "error": {k: [float(f"{v:.4g}") for v in e[idx]] for k, e in err.items()},
        "shape_error": {k: [float(f"{v:.4g}") for v in shape_err(p[idx, :2])] for k, p in preds.items()},
        "shape_metric": "distance from the predicted (x,y) to the nearest point on the true orbit's path: is the rollout on the right orbit, whatever its phase",
        "orbit_scale": {"r_min": float(f"{np.linalg.norm(truth[:, :2], axis=1).min():.4g}"), "r_max": float(f"{np.linalg.norm(truth[:, :2], axis=1).max():.4g}")},
        "final_error": {k: float(f"{e[-1]:.4g}") for k, e in err.items()},
        "median_error_after_400": {k: float(f"{np.median(e[TRAIN_CUTOFF:]):.4g}") for k, e in err.items()},
        "train_seconds": {k: round(v, 1) for k, v in secs.items()},
    }


def control_part():
    from src.control.eval.stats import bootstrap_ci
    R = json.load(open("results/phase_d_pendulum.json")); G = collections.defaultdict(list)
    for r in R: G[r["model"]].append(r)
    out = {}
    for m, rs in G.items():
        st = rs[0]["steps"]; Y = np.array([r["returns"] for r in rs])
        ci = [bootstrap_ci(Y[:, i], n_boot=2000) for i in range(Y.shape[1])]
        final = [np.mean(r["returns"][-3:]) for r in rs]
        out[m] = {"steps": st, "mean": [round(c[0], 1) for c in ci], "lo": [round(c[1], 1) for c in ci],
                  "hi": [round(c[2], 1) for c in ci], "n_seeds": len(rs),
                  "seeds_solved_at_end": int(sum(f > -500 for f in final))}
    return {"system": "pendulum swing-up, MPPI in the learned model (model-based) vs SAC, simulation, noise-free",
            "source": "results/phase_d_pendulum.json", "solved_threshold": -500,
            "solved_definition": "episode return (200 steps) above -500", "models": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-orbit", action="store_true"); a = ap.parse_args()
    path = "results/site_figures.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    if not a.skip_orbit: res["orbit"] = orbit_part()
    res["control"] = control_part()
    json.dump(res, open(path, "w"), indent=1)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    C = {"mlp": "#2a78d6", "piwm": "#eb6834", "pinn": "#1baf7a", "sac": "#2a78d6", "mb": "#eb6834"}
    if "orbit" in res:
        o = res["orbit"]; fig, ax = plt.subplots(figsize=(7, 4))
        for k, lab in [("mlp", "MLP"), ("pinn", "PINN"), ("piwm", "PIWM")]:
            ax.plot(o["steps"], o["error"][k], color=C[k], lw=2, label=lab)
        ax.axvline(TRAIN_CUTOFF, color="#888", lw=1); ax.set_yscale("log")
        ax.set_xlabel("rollout step"); ax.set_ylabel("position error"); ax.legend(); ax.set_title("Two-body orbit: free rollout error")
        plt.savefig("figures/site_orbit_error.png", dpi=160, bbox_inches="tight"); plt.close()
    c = res["control"]["models"]; fig, ax = plt.subplots(figsize=(7, 4))
    for m, d in c.items():
        col = C["sac"] if m == "sac" else C["mb"]
        ax.plot(d["steps"], d["mean"], color=col, lw=2 if m == "sac" else 1.2, label="SAC" if m == "sac" else None)
        if m == "sac": ax.fill_between(d["steps"], d["lo"], d["hi"], color=col, alpha=.15)
    ax.plot([], [], color=C["mb"], label="model-based (6 world models)")
    ax.axhline(-500, color="#888", lw=1, ls="--"); ax.set_xlabel("environment steps"); ax.set_ylabel("episode return")
    ax.legend(); ax.set_title("Pendulum swing-up (5 seeds, mean; SAC with 95% CI)")
    plt.savefig("figures/site_learning_curves.png", dpi=160, bbox_inches="tight"); plt.close()
    print(json.dumps({k: v for k, v in res.get("orbit", {}).items() if k in ("final_error", "median_error_after_400", "train_seconds")}, indent=1))
