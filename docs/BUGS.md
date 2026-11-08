# BUGS
(Each: symptom, how found, root cause, fix, lesson.)

## B1 (Phase A) energy drift above tolerance
Symptom: energy test failed (pendulum 7e-4, acrobot 6.7e-3). Found by: gate test. Suspected an EOM bug;
ruled out by a substep sweep showing 2^5 error reduction per halving (pure RK4 error). Fix: more substeps (D2).
Lesson: check convergence order before suspecting the physics.
## B2 (Phase A) torch/numpy agreement test failed with identical-looking printouts
Cause: test built the action as float32 `torch.tensor([[u]])`, while the state was float64 (precision loss
~1e-8). Fix: dtype=float64 in the test. Lesson: mixed dtype silently loses precision; the sims are float64 throughout.

## B1 (Phase A) energy drift above tolerance
Found by gate test (pendulum 7e-4, acrobot 6.7e-3). Suspected EOM bug; ruled out by a substep sweep with
2^5 reduction per halving (pure RK4 error). Fix: more substeps (D2). Lesson: check convergence order first.
## B2 (Phase A) torch/numpy agreement failed with identical-looking printouts
Test built the action as float32 `torch.tensor([[u]])` against float64 state (precision loss ~1e-8). Fix: float64 everywhere.
## B3 (Phase A) after adding substeps the NumPy-vs-torch trajectory test failed
Test used 1 NumPy RK4 step per control step vs 4 substeps in the sim. Test error, not sim error; test now substeps identically.
Process note: my first commit of Phase A happened with 3 failing tests because a BSD `sed` edit silently failed;
fixed in the next commit. Lesson: run the tests before committing.
