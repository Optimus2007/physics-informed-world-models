"""Write docs/report_control/tables.tex from results/*.json (no hand typed numbers). Negative numbers use math minus."""
import json, collections, numpy as np
from src.control.eval.stats import summarize

J = lambda f: json.load(open(f"results/{f}"))
NAME = {"physics": "Physics", "physics_residual": "Physics + residual", "neural_hamiltonian": "Neural Hamiltonian", "mlp_ensemble": "MLP ensemble",
        "neural_ode": "Neural ODE", "legacy_latent_ode": "Legacy latent ODE", "sac": "SAC", "oracle": "Oracle (true simulator)",
        "physics_nofric": "Physics, no friction term", "physics_nofric_residual": "Physics, no friction term + residual"}
sci = lambda x: ("%.1e" % x).replace("e-0", "e").replace("e-", "e").replace("e+0", "e").replace("e+", "e")
def sci(x):
    m, e = ("%.1e" % x).split("e"); return f"${m}\\times 10^{{{int(e)}}}$"
num = lambda x: f"${x:.0f}$"
ci = lambda s: f"${s['mean']:.0f}$ $[{s['lo']:.0f},\\ {s['hi']:.0f}]$"
out = []

# Table: Phase C
R = J("phase_c_pendulum.json"); G = collections.defaultdict(list)
for r in R: G[(r["model"], r["size"])].append(r)
rows = ["Model & $H=1$ & $H=25$ & $H=100$ & Energy drift, $t=200$ \\\\ \\midrule"]
for m in ["physics", "neural_hamiltonian", "mlp_ensemble", "physics_residual", "neural_ode", "legacy_latent_ode"]:
    rs = G[(m, 100)]; f = lambda k: np.median([r["horizon"][k]["median"] for r in rs])
    rows.append(f"{NAME[m]} & {sci(f('1'))} & {sci(f('25'))} & {sci(f('100'))} & {sci(np.median([r['energy']['200'] for r in rs]))} \\\\")
out.append(("tabCa", "\n".join(rows)))
rows = ["Model & 400 transitions & 1000 transitions & 5000 transitions \\\\ \\midrule"]
for m in ["physics", "neural_hamiltonian", "mlp_ensemble", "physics_residual", "neural_ode", "legacy_latent_ode"]:
    rows.append(NAME[m] + "".join(f" & {sci(np.median([r['horizon']['100']['median'] for r in G[(m, s)]]))}" for s in (8, 20, 100)) + " \\\\")
out.append(("tabCb", "\n".join(rows)))

# Table: Phase D
R = J("phase_d_pendulum.json"); G = collections.defaultdict(list)
for r in R: G[r["model"]].append(r)
rows = ["Method & Return, last three episodes (mean and 95\\% CI) & Seeds solved \\\\ \\midrule"]
for m in ["physics", "physics_residual", "neural_hamiltonian", "mlp_ensemble", "neural_ode", "legacy_latent_ode", "sac"]:
    f = np.array([np.mean(r["returns"][-3:]) for r in G[m]]); s = summarize(f)
    rows.append(f"{NAME[m]} & {ci(s)} & {int((f > -500).sum())} of {len(f)} \\\\")
out.append(("tabD", "\n".join(rows)))

# Table: low data
R = J("phase_d_lowdata_pendulum.json"); G = collections.defaultdict(list)
for r in R: G[(r["model"], r["n_transitions"])].append(r["ret"])
rows = ["Model & 50 transitions & 100 transitions & 200 transitions \\\\ \\midrule"]
for m in ["physics", "physics_residual", "neural_hamiltonian", "mlp_ensemble", "neural_ode", "legacy_latent_ode"]:
    rows.append(NAME[m] + "".join(f" & {sum(v > -500 for v in G[(m, n)])} of {len(G[(m, n)])} ($\\bar R={np.mean(G[(m, n)]):.0f}$)" for n in (50, 100, 200)) + " \\\\")
out.append(("tabL", "\n".join(rows)))

# Table: Phase E, adapted return per shift, plus unadapted and oracle
R = J("phase_e_pendulum.json"); shifts = list(dict.fromkeys(r["shift"] for r in R))
ms = ["physics", "physics_residual", "neural_hamiltonian", "mlp_ensemble", "neural_ode", "legacy_latent_ode"]
rows = ["Shift & Oracle & Unadapted (physics) & " + " & ".join(NAME[m] for m in ms) + " \\\\ \\midrule"]
for s in shifts:
    orc = np.mean([r["ret_oracle"] for r in R if r["model"] == "physics" and r["shift"] == s])
    un = np.mean([r["ret_noadapt"] for r in R if r["model"] == "physics" and r["shift"] == s])
    cells = [num(np.mean([r["ret_adapt"] for r in R if r["model"] == m and r["shift"] == s])) for m in ms]
    rows.append(f"${s.replace(' x', ' \\times ')}$ & {num(orc)} & {num(un)} & " + " & ".join(cells) + " \\\\")
out.append(("tabEa", "\n".join(rows)))
rows = ["Model & 25 & 50 & 100 & 200 & 500 & 1000 \\\\ \\midrule"]
for m in ["physics", "neural_hamiltonian", "mlp_ensemble", "neural_ode", "legacy_latent_ode"]:
    rs = [r for r in R if r["model"] == m and r["shift"] == "g x1.5"]
    rows.append(NAME[m] + "".join(f" & {sci(np.median([r['curve'][n] for r in rs]))}" for n in ["25", "50", "100", "200", "500", "1000"]) + " \\\\")
out.append(("tabEb", "\n".join(rows)))
P = [r for r in R if r["model"] == "physics"]
ea = [abs(r["a"] - r["a_true"]) / r["a_true"] for r in P]; ec = [abs(r["c"] - r["c_true"]) / r["c_true"] for r in P]
for k, v in [("recovAmed", np.median(ea)), ("recovAmax", max(ea)), ("recovCmed", np.median(ec)), ("recovCmax", max(ec))]:
    out.append((k, sci(v)))
out.append(("recovN", str(len(P))))

# Table: Phase F
R = J("phase_f_pendulum.json"); G = collections.defaultdict(list)
for r in R: G[r["model"]].append(r)
rows = ["Model & Return, last three episodes & Prediction error at $H=25$ \\\\ \\midrule"]
for m in ["oracle", "physics_nofric", "physics_nofric_residual", "physics", "neural_hamiltonian", "mlp_ensemble", "neural_ode"]:
    f = np.array([np.mean(r["returns"][-3:]) for r in G[m]]); s = summarize(f)
    if m == "oracle": e = "not applicable"
    else:
        s2 = summarize([r["err"]["25"]["median"] for r in G[m]]); e = f"{sci(s2['mean'])} $[{sci(s2['lo'])[1:-1]},\\ {sci(s2['hi'])[1:-1]}]$"
    rows.append(f"{NAME[m]} & {ci(s)} & {e} \\\\")
out.append(("tabF", "\n".join(rows)))
open("docs/report_control/tables.tex", "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" if k.startswith("recov") else f"\\newcommand{{\\{k}}}{{%\n{v}%\n}}\n" for k, v in out))
print("tables written")
