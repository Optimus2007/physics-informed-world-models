"""Turn results/*.json into figures/*.{pdf,png} and results/summary_table.md. Never hand-edited numbers."""
import json, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.control.eval.stats import summarize, bootstrap_ci

os.makedirs("figures", exist_ok=True)
C = {"physics": "#0072B2", "physics_residual": "#56B4E9", "neural_hamiltonian": "#009E73", "mlp_ensemble": "#E69F00",
     "neural_ode": "#CC79A7", "legacy_latent_ode": "#999999", "sac": "#D55E00", "oracle": "#000000",
     "physics_nofric": "#0072B2", "physics_nofric_residual": "#56B4E9"}
LBL = {k: k.replace("_", " ") for k in C}
J = lambda f: json.load(open(f"results/{f}"))
save = lambda name: (plt.savefig(f"figures/{name}.png", dpi=160, bbox_inches="tight"), plt.savefig(f"figures/{name}.pdf", bbox_inches="tight"), plt.close())
tab = []


def fig_learning_curves():
    R = J("phase_d_pendulum.json"); G = collections.defaultdict(list)
    for r in R: G[r["model"]].append(r)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m, rs in G.items():
        st = np.array(rs[0]["steps"]); Y = np.array([r["returns"] for r in rs])
        ci = np.array([bootstrap_ci(Y[:, i], n_boot=2000)[1:] for i in range(Y.shape[1])])
        for a in ax:
            a.plot(st, Y.mean(0), color=C[m], label=LBL[m], lw=2 if m == "sac" else 1.5, ls="--" if m != "sac" else "-")
            a.fill_between(st, ci[:, 0], ci[:, 1], color=C[m], alpha=0.15)
    ax[0].set_xlim(0, 10000); ax[0].set_title("All methods (SAC needs about 10 times more steps)")
    ax[1].set_xlim(400, 2200); ax[1].set_ylim(-700, -250); ax[1].set_title("Zoom on the model based agents")
    for a in ax: a.set_xlabel("environment steps"); a.set_ylabel("episode return (200 steps)")
    ax[0].legend(fontsize=7); save("fig1_learning_curves")
    for m, rs in G.items():
        f = np.array([np.mean(r["returns"][-3:]) for r in rs]); s = summarize(f)
        tab.append(f"| D: final return (last 3 eps) | {m} | {s['mean']:.0f} [{s['lo']:.0f}, {s['hi']:.0f}] | IQM {s['iqm']:.0f}, n={s['n']} |")


def fig_lowdata():
    R = J("phase_d_lowdata_pendulum.json"); G = collections.defaultdict(list)
    for r in R: G[(r["model"], r["n_transitions"])].append(r["ret"])
    ns = sorted({k[1] for k in G}); ms = ["physics", "physics_residual", "neural_hamiltonian", "mlp_ensemble", "neural_ode", "legacy_latent_ode"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(ms):
        y = [np.mean([v > -500 for v in G[(m, n)]]) for n in ns]
        ax.plot(np.arange(len(ns)) + (i - 2.5) * 0.04, y, "o-", color=C[m], label=LBL[m])
    ax.set_xticks(range(len(ns))); ax.set_xticklabels(ns); ax.set_xlabel("random transitions used to fit the model"); ax.set_ylabel("fraction of seeds that swing up (n=5)")
    ax.legend(fontsize=7); save("fig2_lowdata")
    for m in ms:
        for n in ns:
            v = G[(m, n)]; s = summarize(v); tab.append(f"| H1 low-data return @{n} | {m} | {s['mean']:.0f} [{s['lo']:.0f}, {s['hi']:.0f}] | solved {sum(x > -500 for x in v)}/{len(v)} |")


def fig_c():
    R = J("phase_c_pendulum.json"); G = collections.defaultdict(list)
    for r in R: G[(r["model"], r["size"])].append(r)
    ms = ["physics", "physics_residual", "neural_hamiltonian", "mlp_ensemble", "neural_ode", "legacy_latent_ode"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m in ms:
        hs = [1, 10, 25, 50, 100]
        y = [np.median([r["horizon"][str(h)]["median"] for r in G[(m, 100)]]) for h in hs]
        ax[0].loglog(hs, y, "o-", color=C[m], label=LBL[m])
        ts = ["10", "50", "100", "200"]
        ax[1].semilogy([int(t) for t in ts], [np.median([r["energy"][t] for r in G[(m, 100)]]) for t in ts], "o-", color=C[m])
    ax[0].set_xlabel("rollout horizon (steps of 0.05 s)"); ax[0].set_ylabel("normalised RMSE (median over seeds)"); ax[0].legend(fontsize=7)
    ax[0].set_title("Prediction error vs horizon (5000 transitions)")
    ax[1].set_xlabel("time (steps), u=0"); ax[1].set_ylabel(r"$|E(x_t)-E(x_0)|\,/\,(|E_0|+1)$"); ax[1].set_title("Energy drift of model rollouts")
    save("fig3_horizon_energy")
    for m in ms:
        for sz in (8, 100):
            v = [r["horizon"]["100"]["median"] for r in G[(m, sz)]]; s = summarize(v)
            tab.append(f"| C: H=100 error, {sz*50} transitions | {m} | {s['mean']:.2e} [{s['lo']:.2e}, {s['hi']:.2e}] | n={s['n']} |")


def fig_e():
    R = J("phase_e_pendulum.json"); shifts = list(dict.fromkeys(r["shift"] for r in R))
    ms = ["physics", "neural_hamiltonian", "mlp_ensemble", "neural_ode", "legacy_latent_ode"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    w = 0.14
    for k, (key, ttl) in enumerate([("ret_noadapt", "No adaptation"), ("ret_adapt", "Adapted with 50 transitions")]):
        ax = axes[k]
        for i, m in enumerate(ms):
            y = []; lo = []; hi = []
            for s in shifts:
                v = [r[key] for r in R if r["model"] == m and r["shift"] == s]; mm, l, h = bootstrap_ci(v, n_boot=2000)
                y.append(mm); lo.append(mm - l); hi.append(h - mm)
            ax.bar(np.arange(len(shifts)) + (i - 2) * w, y, w, yerr=[lo, hi], color=C[m], label=LBL[m], capsize=1.5)
        orc = [np.mean([r["ret_oracle"] for r in R if r["model"] == "physics" and r["shift"] == s]) for s in shifts]
        ax.plot(np.arange(len(shifts)), orc, "k_", ms=22, mew=2, label="oracle (true sim)")
        ax.set_xticks(range(len(shifts))); ax.set_xticklabels(shifts, rotation=30); ax.set_ylabel("episode return"); ax.set_title(ttl)
    axes[0].legend(fontsize=7); save("fig4_shift_returns")
    for s in shifts:
        for m in ["physics", "physics_residual", "neural_hamiltonian", "mlp_ensemble", "neural_ode", "legacy_latent_ode"]:
            va = [r["ret_adapt"] for r in R if r["model"] == m and r["shift"] == s]; vn = [r["ret_noadapt"] for r in R if r["model"] == m and r["shift"] == s]
            sa, sn = summarize(va), summarize(vn)
            tab.append(f"| E: return {s} | {m} | adapt {sa['mean']:.0f} [{sa['lo']:.0f}, {sa['hi']:.0f}]; no-adapt {sn['mean']:.0f} [{sn['lo']:.0f}, {sn['hi']:.0f}] | n={sa['n']} |")
    # transitions needed
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, s in zip(axes, ["g x1.5", "l x1.5"]):
        for m in ["physics", "neural_hamiltonian", "mlp_ensemble", "neural_ode", "legacy_latent_ode"]:
            rs = [r for r in R if r["model"] == m and r["shift"] == s]; ns = sorted(int(n) for n in rs[0]["curve"])
            ax.loglog(ns, [np.median([r["curve"][str(n)] for r in rs]) for n in ns], "o-", color=C[m], label=LBL[m])
        ax.set_title(f"Shift {s}"); ax.set_xlabel("new transitions used for adaptation"); ax.set_ylabel("H=25 error in shifted env")
    axes[0].legend(fontsize=7); save("fig5_transitions_needed")
    # parameter recovery (identifiable combos)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    P = [r for r in R if r["model"] == "physics"]
    for a, (k, kt, t) in zip(ax, [("a", "a_true", "a = 3g/2l"), ("c", "c_true", "c = 3/(m l²)")]):
        a.scatter([r[kt] for r in P], [r[k] for r in P], s=14, color=C["physics"]); lim = [0, max(r[kt] for r in P) * 1.1]
        a.plot(lim, lim, "k--", lw=1); a.set_xlabel(f"true {t}"); a.set_ylabel(f"estimated {t}")
    save("fig6_param_recovery")
    err = [abs(r["a"] - r["a_true"]) / r["a_true"] for r in P]; err2 = [abs(r["c"] - r["c_true"]) / r["c_true"] for r in P]
    tab.append(f"| E: rel. error of a / c after 50 transitions | physics | median {np.median(err):.1e} / {np.median(err2):.1e}; max {max(err):.1e} / {max(err2):.1e} | n={len(P)} |")
    # NOTE: individual g/m/l estimates are deliberately NOT tabulated: with learned m,l they are not identifiable (DECISIONS D1/D4).


def fig_f():
    if not os.path.exists("results/phase_f_pendulum.json"): return
    R = J("phase_f_pendulum.json"); G = collections.defaultdict(list)
    for r in R: G[r["model"]].append(r)
    ms = ["oracle", "physics", "physics_nofric_residual", "physics_nofric", "neural_hamiltonian", "mlp_ensemble", "neural_ode"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, m in enumerate(ms):
        f = np.array([np.mean(r["returns"][-3:]) for r in G[m]]); s = summarize(f)
        ax[0].bar(i, s["mean"], color=C.get(m, "#777"), yerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]], capsize=3)
        tab.append(f"| F: return (last 3 eps), real env | {m} | {s['mean']:.0f} [{s['lo']:.0f}, {s['hi']:.0f}] | IQM {s['iqm']:.0f}, n={s['n']} |")
        if m != "oracle":
            e = [r["err"]["25"]["median"] for r in G[m]]; s2 = summarize(e)
            ax[1].bar(i, s2["mean"], color=C[m], yerr=[[s2["mean"] - s2["lo"]], [s2["hi"] - s2["mean"]]], capsize=3)
            tab.append(f"| F: H=25 error, real env | {m} | {s2['mean']:.2e} [{s2['lo']:.2e}, {s2['hi']:.2e}] | n={s2['n']} |")
    for a in ax: a.set_xticks(range(len(ms))); a.set_xticklabels([LBL.get(m, m) for m in ms], rotation=35, ha="right", fontsize=8)
    ax[0].set_ylabel("episode return"); ax[0].set_title("Real pendulum (friction + hidden motor limit)")
    ax[1].set_ylabel("H=25 prediction error"); ax[1].set_yscale("log"); save("fig7_friction_failure")


if __name__ == "__main__":
    for f in (fig_learning_curves, fig_lowdata, fig_c, fig_e, fig_f): f()
    open("results/summary_table.md", "w").write("| experiment | model | mean [95% bootstrap CI] | notes |\n|---|---|---|---|\n" + "\n".join(tab) + "\n")
    print("ok", len(tab), "rows")
