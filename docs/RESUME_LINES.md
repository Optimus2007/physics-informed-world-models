# Resume lines (every number is in `results/`; do not add numbers that are not)

- Built a reproducible PyTorch study of physics-structured vs black-box world models for model-predictive control on a swing-up pendulum (batched RK4 simulators, own MPPI planner cross-checked against `pytorch_mppi`, 38 unit tests, 5 seeds with bootstrap CIs).
- Showed that re-estimating physical parameters from 50 transitions after a gravity/length/mass shift reduces 25-step prediction error to about 1e-10 (noise-free simulation), versus 3e-4 to 1e-2 for neural-ODE/MLP/Hamiltonian models even with 1000 transitions, while control return after adaptation was within seed noise of the best black-box models.
- Implemented a SymODEN-style neural Hamiltonian with control: 100-step prediction error 6.7e-4 (vs 4.9e-3 MLP ensemble, 2.0e-2 neural ODE) with flat energy drift; also reported negative results (no sample-efficiency gap at 500 transitions; residual hurts on clean data; physics-model failure under unmodelled friction fixed only to MLP level).
- Compared against SAC: model-based agents solved the task after ~500 environment steps, SAC needed ~8-10k (3 of 5 seeds solved at 10k).

**README blurb (4 lines)**
Control extension of physics-informed world models: pendulum swing-up with MPPI in learned models.
Compares structured physics, neural-Hamiltonian (SymODEN-style), MLP-ensemble, neural-ODE and legacy latent-ODE models plus SAC.
Tests data efficiency, adaptation to changed gravity/length/mass from 50 transitions, and failure under unmodelled friction.
Simulation-only, 5 seeds; includes negative results; see docs/LEARNING_REPORT.md.
