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

## B4 (Phase C/F design) friction parameter initialised from the env's true value
Found in code review before the Phase E/F runs: PhysicsStructured defaults init to env.params, so b would start at the true friction in
Phase F (leak). Fix: experiments/phase_c.make sets b init = 0 always. Phase C/D runs were unaffected (b=0 there).
## B5 (Phase A-test) IQM test passed vacuously twice
Outliers were symmetric (cancelled in the mean) and then my replace-string did not match. Lesson: a test that cannot fail is not a test; check it fails on a broken implementation.
## B6 (analysis) misleading 'g estimate' rows
Table rows compared estimated g with the true shifted g although the model's learned m,l are on the non-identifiable manifold (e.g. l=0.78), so g is expected to differ.
Removed; the valid metrics are the identifiable combinations a=3g/2l, c=3/(m l^2) (rel. error ~1e-10).
## B7 (Phase C) hanging pytest: n/a. Note: Python `sleep`-polling is blocked in this environment; background until-loops used.
