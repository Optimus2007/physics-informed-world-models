# DECISIONS
(Running log: choice, alternatives, why.)

## D0 Environment (2026-11-08)
Python 3.13.9, numpy 2.3.5, torch 2.12.0, scipy 1.16.3. Installed via pip: torchdiffeq 0.2.5,
gymnasium 1.4.0, pytorch_mppi 0.9.1, rliable 1.2.0. Hardware: laptop CPU (macOS).
Code is imported as `src.control.*` from repo root (no packaging change to the old src/).

## D1 Environments and Lagrangian derivations (Phase A)
- Pendulum: uniform rod (I = m l^2/3) so the EOM equals Gymnasium Pendulum-v1 (verified numerically in
  tests/test_envs.py::test_pendulum_matches_gymnasium_dynamics). theta=0 is upright. We do NOT step the
  Gymnasium env itself in experiments (it uses Euler, float32, non-batched, and fixed 200-step episodes);
  instead our batched torch RK4 sim is ground truth and Gymnasium is an independent cross-check + a
  parameter-shift wrapper (`wrappers.make_gym_pendulum`). Derivations are in each env file's docstring
  (pendulum, cartpole: point-mass pole, acrobot: point-mass links).
- Cartpole/acrobot are own simulators (Gymnasium's are discrete-action / fixed-parameter-ish).
- Identifiability (important later): the pendulum acceleration is (3g/2l) sin(th) + 3u/(m l^2) - 3b w/(m l^2).
  Only the combinations a=3g/(2l), c=3/(m l^2), d=3b/(m l^2) affect trajectories, so (m,l,g) are NOT
  separately identifiable from state-action data; two constraints for three unknowns.

## D2 Integrator and energy tolerance
Gate A tolerance: max |E(t)-E(0)| / max(1,|E0|) < 1e-4 over 1000 steps (50 s) with u=0, b=0.
With 1 RK4 substep per control step (dt=0.05) the drift was 1.4e-3 (pendulum), 1.1 (acrobot, chaotic).
Halving the step reduced drift by ~32x each time, i.e. pure RK4 truncation, not an EOM bug. Chosen
substeps: pendulum 4, cartpole 4, acrobot 8 (control dt stays 0.05 s). Cost: ~4-8x more RHS evals per step.
Control inputs are zero-order-hold over the control interval.

## D1 Environments and Lagrangian derivations (Phase A)
- Pendulum: uniform rod (I = m l^2/3) so the EOM equals Gymnasium Pendulum-v1 (verified in
  tests/test_envs.py::test_pendulum_matches_gymnasium_dynamics). theta=0 upright. Our batched torch RK4 sim
  is ground truth; Gymnasium is an independent cross-check + parameter-shift wrapper (`wrappers.py`), since
  its env is Euler, float32, unbatched, 200-step episodes. Derivations are in each env file's docstring.
- Cartpole/acrobot: own simulators (point-mass pole / point-mass links).
- Identifiability: pendulum accel = (3g/2l) sin(th) + 3u/(m l^2) - 3b w/(m l^2). Only a=3g/2l, c=3/(m l^2),
  d=3b/(m l^2) affect trajectories, so (m,l,g) are NOT separately identifiable from state-action data.

## D2 Integrator and energy tolerance
Gate A tolerance: max|E(t)-E(0)|/max(1,|E0|) < 1e-4 over 1000 steps (50 s), u=0, b=0.
At 1 RK4 substep per control step (dt=0.05) drift was 1.4e-3 (pendulum) and 1.1 (acrobot, chaotic).
Halving the step cut drift ~32x each time, i.e. pure RK4 truncation, not an EOM bug. Chosen substeps:
pendulum 4, cartpole 4, acrobot 8 (control dt stays 0.05 s, zero-order-hold actions).

## D3 Planner settings that work with the TRUE simulator (Gate B; results/gate_b_*.json)
- Pendulum (umax=2, hanging at rest start): MPPI H=40, N=500, lambda=0.5, sigma=1.0 -> 5/5 seeds swing up and balance
  (|theta|<0.3 over last 50 of 200 steps); ~2.6 s per 200-step episode. CEM (H=40,N=500,4 iters,50 elites) 5/5, ~10 s.
- Cartpole (M=1, m=0.3, l=0.5, umax=10): MPPI H=50, N=1000, lambda=0.5, sigma=4.0 -> 5/5 (300 steps), but returns vary
  widely (-363 to -811) = swing-up speed differs. 
- Own MPPI vs pytorch_mppi (same H,N,lambda,sigma; results/crosscheck_mppi.json): mean return own -358.0 vs lib -373.4
  over 5 seeds. Same order, no material disagreement; difference is within seed spread.
- Design: the nominal unperturbed sequence is always one candidate; post-clip perturbations are used in the update;
  no terminal cost (reward over horizon only); acrobot is implemented/tested but not planned for (time budget).

## D4 World-model design (Phase C)
- Common training: one-step MSE on increments normalised by per-dim std of (x_next - x), Adam lr 1e-3, batch 256, <=1500 steps,
  early stopping on a validation split BY EPISODE (20%), best-val weights restored. Identical for all models (fairness).
  No multi-step (unrolled) training loss for any model; so long-horizon quality reflects model class, not a better loss.
- MLP ensemble: K=5 members (3x200 SiLU), Poisson(1) bootstrap weights, MSE not Gaussian NLL (simplest valid choice; the spec allows MSE);
  planning uses the ensemble mean. Inputs: (cos,sin) for angles, standardised features.
- Neural ODE: dx/dt = MLP(features, u/umax) * (dstd/dt), RK4, 2 substeps. Output layer not zero-init.
- Structured physics: true analytic RHS (envs/*.py pure functions) with exp-parametrised (m,l,g,b); initialised at truth x U(0.6,1.4)
  per seed (b initialised at 0 - never from the env). Same RK4 sub-stepping as the sim (so zero model error is attainable).
  Residual: MLP(64,64) with 1e-3-scaled output layer + L2 penalty 1e-2 on its contribution (APHYNITY-flavoured, not APHYNITY's constrained opt).
- Neural Hamiltonian (SymODEN-style): V(q), Minv(q)=LL^T+eps (L from MLP), g(q); data in (q,qdot) converted with the model's own mass
  matrix. No dissipation term. One bias of V is unidentifiable by construction (a constant in V).
  Honest differences from Zhong et al.: no known-structure shortcuts (no fixed mass matrix), no angle-embedding trick besides (cos,sin).
- Legacy latent ODE: see D5.
- Identifiability (pendulum): with b=0, only a=3g/2l and c=3/(m l^2) are identifiable. Verified: tests/test_models.py
  recovers a, c to 2% and g to 2% when m,l are held fixed; individual (m,l,g) are NOT recovered (e.g. 1.66, 0.776, 7.76 vs 1,1,10).
- Evaluation: horizon RMSE normalised by per-dim std of true trajectories, angles wrapped; median and mean over 200 held-out wide-init
  random-action trajectories. Energy drift: TRUE energy evaluated on MODEL states, u=0, median over 50 starts.

## D5 Legacy latent ODE baseline
Existing PhysicsInformedWorldModel kept as encoder->dopri5 ODE->decoder (tanh, hidden 64, latent 4), changes: action input to ODE,
residual decoding x + dstd*(dec(z1)-dec(z0)) + dmean, orbit energy penalty dropped. It was never Hamiltonian.
It is expected to be weak: it was designed to fit ONE trajectory, and has 64 hidden units vs 200 for others.

## Gate C result (results/phase_c_pendulum.json, 5 seeds x 3 data sizes)
Median H=100 error (400 / 1000 / 5000 transitions): physics 1.0e-4/5.8e-4/1.1e-4; neural_hamiltonian 9.1e-4/8.2e-4/6.7e-4;
mlp_ensemble 1.0e-2/9.5e-3/4.9e-3; neural_ode 4.3e-2/2.0e-2/2.0e-2; legacy 1.4e-1/7.2e-2/5.9e-2. Energy drift at t=200:
Hamiltonian ~7e-5 (flat), physics 4e-6..6e-4, MLP 4e-3..1.4e-2, NODE 1e-2..4e-2. The residual hurts on clean data
(H=25: 1.7e-3 vs 2.2e-5 at 400 transitions) because it adds capacity without benefit; reported as is.
Caveat: the data is random-action, wide-init; the same models are used in control where state distribution differs.

## Gate D result (results/phase_d_pendulum.json; 5 seeds, 10 MBRL iterations of 200 steps after 500 random steps)
ALL six world models (incl. legacy) solve the swing-up from the very first planning episode (returns -370..-400, 5/5 seeds,
true-model MPPI gives about -320..-372). So with 500 random transitions the task does not discriminate models: H1 is NOT
supported or refuted by this experiment. SAC (same env, 10k steps): median return -1897 at 0, -1050 at 4k, -383 at 8k;
3/5 seeds solved at 10k (last-3-eval mean -373,-369,-378 vs -754,-735 for the others). MBRL is ~>10x more sample efficient than SAC
here, regardless of model class. Returns are almost identical across models because they share start states and planner seeds.
Wall time per run: physics 109 s, MLP 1733 s, NODE 2480 s, legacy 2257 s, NH 8762 s (CPU, 1 thread, contended).
Follow-up: low-data sweep (50-200 random transitions) in experiments/phase_d_lowdata.py to actually test H1.
Note: no speed clamp in sim -> failed swings can give returns below -3000 (e.g. SAC -13176).

## H1 low-data result (results/phase_d_lowdata_pendulum.json; 5 seeds, one planning episode after k random episodes)
Solved (return > -500) at 50/100/200 random transitions: physics 5/5,5/5,5/5; physics+residual 5/5,5/5,5/5; neural_hamiltonian 5/5,5/5,5/5;
mlp_ensemble 3/5,2/5,5/5; neural_ode 4/5,2/5,5/5; legacy 1/5,0/5,4/5. Non-monotone (100 worse than 50) for the black boxes =
high variance with n=5 seeds, a data-coverage lottery with 1-2 episodes; do not over-read. Supports H1 qualitatively in the
50-100 transition regime for pendulum only; no effect at >=500 transitions.

## Phase E result (results/phase_e_pendulum.json; 5 seeds x 7 shifts x 6 models; 50 new transitions = 5 episodes x 10 steps)
- Prediction (H=25 median error on shifted env): adapting physics params gives ~1e-10 (exact, in sim); physics+residual 1e-4;
  neural_hamiltonian 2e-3..1e-2; black boxes 1e-2..9e-2 (before adaptation 0.05..0.47 for all).
- Control (median return, per-shift): after adaptation EVERY model is within a few % of the oracle (true shifted simulator MPPI) for
  most shifts, e.g. g x1.5: oracle -434, physics -438, MLP -449, NODE -493, legacy -480. Control return is far less sensitive than
  prediction error: MPC replans every step and corrects model errors with feedback. So H2 is strongly supported for MODEL ACCURACY
  and only weakly (small gaps, mostly inside seed noise; see analysis CIs) for CONTROL RETURN.
- Hard shifts (g x2.0, l x1.5, m x2.0): the oracle itself scores -540..-620 (slow swing-up within 200 steps / torque limit 2), so the
  '-500 = solved' threshold used elsewhere is not meaningful there; compare against the oracle.
- Unadapted: gap from oracle large for g x1.5/x2.0 (-1270..-1500). Some unadapted black boxes beat physics on g x0.5 (MLP -287 vs physics -599)
  by luck of an easier regime; no systematic claim.
- 'ret_adapt_known' (adapt only the truly shifted parameter) also stored for physics.

## Phase F result (results/phase_f_pendulum.json; real pendulum b=0.15, hidden actuator saturation 1.5 < command range 2; 5 seeds, 10 MBRL iters)
H=25 prediction error on the real env (mean [95% CI], results/summary_table.md): physics_nofric 0.279 [0.276,0.281]; neural_hamiltonian 0.265;
physics_nofric_residual 0.0062 [0.0041,0.0085]; mlp_ensemble 0.0071; neural_ode 0.0086; physics with learnable b 0.081 [0.033,0.129] (friction learned, saturation not).
Return (last 3 episodes): oracle -640, physics -631, nofric_residual -639, nofric -673 [-724,-625], NH -638, MLP -638, NODE -644.
=> H4 supported for PREDICTION (residual is ~45x better than the no-friction physics model) but the residual only reaches MLP-level accuracy, it does not beat black boxes;
for CONTROL the MPC feedback hides most of the model error (only no-friction physics is slightly, not significantly, worse: CI overlaps).
Scope not done: cartpole/acrobot learned-model experiments (simulators + true-model planning gate only), because of the time budget.
