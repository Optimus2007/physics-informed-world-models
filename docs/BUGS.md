# BUGS
(Each: symptom, how found, root cause, fix, lesson.)

## B1 (Phase A) energy drift above tolerance
Symptom: energy test failed (pendulum 7e-4, acrobot 6.7e-3). Found by: gate test. Suspected an EOM bug;
ruled out by a substep sweep showing 2^5 error reduction per halving (pure RK4 error). Fix: more substeps (D2).
Lesson: check convergence order before suspecting the physics.
## B2 (Phase A) torch/numpy agreement test failed with identical-looking printouts
Cause: test built the action as float32 `torch.tensor([[u]])`, while the state was float64 (precision loss
~1e-8). Fix: dtype=float64 in the test. Lesson: mixed dtype silently loses precision; the sims are float64 throughout.
