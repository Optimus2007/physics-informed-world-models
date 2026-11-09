# Physics-informed world models for control: one-page summary

**Question.** Does building physics structure into a learned world model help a model-based agent learn control with less data, and keep working when
gravity, length or mass change at test time by re-estimating a few parameters from ~50 transitions?

**Method.** Pendulum swing-up (own batched RK4 simulator, cross-checked against Gymnasium Pendulum-v1). Model-predictive control with a batched MPPI planner
(cross-checked against `pytorch_mppi`). Six world models trained identically: MLP ensemble, neural ODE, structured physics (+/- neural residual),
SymODEN-style neural Hamiltonian with control, and the repo's earlier latent-ODE adapted to take actions. SAC as a model-free reference.
5 seeds per configuration; 95% bootstrap CIs. Simulation only, noise-free, state-based.

**Key results** (all in `results/`; details and caveats in `docs/LEARNING_REPORT.md`)
1. *Prediction and energy:* median error after 100 steps (5000 training transitions): physics 1.1e-4, neural Hamiltonian 6.7e-4, MLP ensemble 4.9e-3, neural ODE 2.0e-2.
   The Hamiltonian's energy drift stays flat (~7e-5 over 200 steps) vs 4e-3 (MLP) and 3e-2 (neural ODE). (`phase_c_pendulum.json`)
2. *Adaptation:* after a gravity/length/mass shift, 50 new transitions bring every model close to an oracle planner's return (e.g. g x1.5: unadapted about -1260, adapted
   physics -443 [-461,-431], MLP -456 [-474,-438], oracle about -434). The physics model reaches ~1e-10 prediction error; no black-box model matched that even with 1000 transitions.
   (`phase_e_pendulum.json`) Return differences between models after adaptation are small and their CIs overlap.
3. *Sample efficiency:* with 500 random transitions every model (even the legacy one) solves the task, so no gap there; with only 50 transitions physics, physics+residual
   and the Hamiltonian solved 5/5 seeds vs MLP 3/5, neural ODE 4/5, legacy 1/5 (suggestive only). Model-based agents needed about 500 steps; SAC needed ~8-10k (3/5 seeds solved at 10k).
   (`phase_d_*.json`)
4. *Where physics fails:* with unmodelled friction + motor limit, the physics model's H=25 error rises to 0.28; a learned residual restores 0.006 (MLP level, not better). Control return barely changes. (`phase_f_pendulum.json`)

**Key figure.** `figures/fig5_transitions_needed.png` (prediction error vs number of adaptation transitions) and `figures/fig4_shift_returns.png`.

**Limitations.** Pendulum only for learned-model experiments (cartpole/acrobot simulators exist, not used for learning); noise-free simulation with exactly right model structure
(best case for the physics model); 5 seeds; MPC feedback hides model error in return metrics; only (a = 3g/2l, c = 3/(ml^2)) are identifiable, not m, l, g separately;
one post-hoc experiment (low-data sweep) was added after seeing that the main sweep did not discriminate; the `--fast` reproduction mode is untimed.
Not new research: it re-implements known ideas (SymODEN, PETS-style ensembles, MPPI) as a careful small comparison.
