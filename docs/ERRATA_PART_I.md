# Corrections to the Part I report (`report.pdf`)

`report.pdf` is kept as written. Some of its statements do not match the code in `src/` or a seeded re-run of it. Read it with these corrections.

| Report says | What the code does / what a re-run shows |
|---|---|
| "I only let them train on the first 20% of the data" | Only the PINN trains on that much data (the first 2,000 steps). The PIWM trains on the first 400 steps (4%); the naive MLP is fitted on one-step pairs from the whole trajectory (`src/train_naive.py`). |
| A "strict thermodynamic rule: the total energy (Hamiltonian) of the system must remain completely constant" | The energy term is a soft penalty: 0.1 × the variance of the model's energy estimate along the training window (`src/train_world_model.py`). It is not a hard constraint and the model is not Hamiltonian. |
| The world model "perfectly conserves energy" and stays "tightly locked to the true physics" for the remaining 9,600 steps | Energy was not measured numerically. In a seeded re-run (`analysis/make_site_figures.py`, seed 0) the PIWM stays within about 0.24 of the true orbit's path for all 10,000 steps, but its period is slightly off, so its position error passes 0.5 by step 580. |
| "25x extrapolation factor … compared to standard methods" | 25× is the rollout length (10,000 steps) over the PIWM's training window (400 steps). It is not a measured advantage over the baselines: they were trained on different amounts of data, and in the re-run the MLP stays closer to the true path. |
| The naive MLP "spiraled inward" because it does not understand energy conservation | True of the original, unseeded run in `assets/`. In the seeded re-run the MLP stays within 0.017 of the orbit's path but drifts in phase. The cause of the spiral was not tested. |

The training scripts set no random seed, so the figures in `assets/` and in the report come from single unseeded runs.
