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
