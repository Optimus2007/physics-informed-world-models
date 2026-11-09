"""One-command reproduction. `--fast` = reduced seeds/iterations (NOT yet timed end-to-end; see report_control.pdf).
Wall-clock of the full run was NOT recorded; phase_d alone is ~83k CPU-seconds (see results/phase_d_pendulum.json)."""
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
