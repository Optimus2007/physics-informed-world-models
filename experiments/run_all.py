"""One-command reproduction. `--fast` = reduced seeds/iterations (NOT yet timed end-to-end; see docs/LEARNING_REPORT.md).
Full run on an 8-core laptop took roughly: phase_c ~30 min, phase_d ~4 h (neural_hamiltonian dominates), low-data ~1 h, phase_e ~1 h, phase_f ~1.5 h."""
import subprocess, sys
fast = "--fast" in sys.argv
S = "2" if fast else "5"
steps = [["-m", "experiments.phase_c", "--seeds", S] + (["--sizes", "8", "20"] if fast else []),
         ["-m", "experiments.phase_d", "--seeds", S, "--iters", "4" if fast else "10", "--sac_steps", "4000" if fast else "10000"],
         ["-m", "experiments.phase_d_lowdata", "--seeds", S],
         ["-m", "experiments.phase_e", "--seeds", S],
         ["-m", "experiments.phase_f", "--seeds", S, "--iters", "4" if fast else "10"],
         ["-m", "analysis.make_figures"]]
for s in steps:
    print("RUN", " ".join(s), flush=True); subprocess.check_call([sys.executable] + s)
