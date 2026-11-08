# DECISIONS
(Running log: choice, alternatives, why.)

## D0 Environment (2026-11-08)
Python 3.13.9, numpy 2.3.5, torch 2.12.0, scipy 1.16.3. Installed via pip: torchdiffeq 0.2.5,
gymnasium 1.4.0, pytorch_mppi 0.9.1, rliable 1.2.0. Hardware: laptop CPU (macOS).
Code is imported as `src.control.*` from repo root (no packaging change to the old src/).
