# Physics-Informed World Models for Learning Dynamical Systems

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Research_Complete-success.svg)

This repository contains the official codebase and research report for **Physics-Informed World Models (PIWM)**.

The project investigates the catastrophic failures of standard deep learning models when predicting continuous dynamical systems (specifically orbital mechanics) and implements a synthesized architecture utilizing **Latent Neural Ordinary Differential Equations (ODEs)** constrained by strict thermodynamic priors to achieve stable, infinite-horizon extrapolation.

📄 **Read the full research paper:** [`report.pdf`](./report.pdf)

---

## 🔬 Overview & Key Findings

Predicting complex physical environments is a cornerstone of scientific computing and Model-Based Reinforcement Learning (MBRL). Traditional numerical solvers (like Runge-Kutta) are accurate but computationally expensive, driving the need for Machine Learning surrogates.

However, pure neural networks lack thermodynamic priors, resulting in **compounding rollout errors**. Standard Physics-Informed Neural Networks (PINNs) mitigate this but suffer from **spectral bias** and phase drift over extended timelines due to a lack of strict causality.

**This project demonstrates:**
1. **The Failure of Pure ML:** Standard autoregressive models fundamentally fail to conserve energy.
2. **The Limits of PINNs:** Standard continuous-time PINNs lose phase alignment over long horizons.
3. **The Success of PIWMs:** By compressing reality into a causal latent space and substituting discrete neural layers with an adaptive Neural ODE solver (`dopri5`), model achieved a **25x extrapolation factor** without breaking energy conservation.

---

## 📊 Experimental Results

All models were trained on only the first **20% (400 steps)** of a simulated 2-body orbital trajectory and forced to predict the remaining **80% (9,600 steps)** zero-shot.

### 1. The Ground Truth (RK4 Integrator)
The target system is an elliptical 2-body orbit governed by Newtonian gravity, generated using a custom RK4 numerical solver.
See [assets](assets) for the orbit plot and other generated figures.

### 2. Baseline 1: Naive Autoregressive MLP
Trained purely on data, the naive model rapidly destabilizes during extrapolation. Lacking an energy-conservation prior, the compounding errors alter the implied momentum, causing a catastrophic unphysical spiral.
See [assets](assets) for the baseline visualization.

### 3. Baseline 2: Standard Physics-Informed Neural Network (PINN)
The PINN enforces gravitational constraints via Automatic Differentiation. While locally constrained by physics, the continuous mapping of time to space fails to close the periodic orbit over a 10,000-step horizon due to spectral bias.
See [assets](assets) for the PINN visualization.

### 4. Physics-Informed World Model (PIWM)
The Latent Neural ODE World Model vastly outperforms the baselines. By combining adaptive ODE integration with a Hamiltonian energy-variance loss, the trajectory remains tightly mathematically bounded to the ground-truth simulation without drift or spectral amnesia.
See [assets](assets) for the world model visualization.

---

## ⚙️ Repository Structure

A clean, modular architecture built for reproducibility and easy expansion into MBRL environments.

```text
physics-informed-world-models/
├── assets/                 # Versioned figure assets
│   ├── naive_mlp_failure.png
│   ├── orbit_plot.png
│   ├── pinn_prediction.png
│   └── world_model_success.png
├── data/                   # Data generation scripts and local artifacts
│   ├── generate_orbit.py   # RK4 ground-truth simulator
│   └── trajectories/       # Generated .npy outputs (gitignored)
│       ├── naive_mlp_prediction.npy
│       └── orbit_data.npy
├── src/                    # Core machine learning architecture
│   ├── dynamics.py         # True mathematical derivatives
│   ├── models.py           # PIWM Autoencoder & Neural ODE definitions
│   ├── pinns.py            # PINN architecture and physical loss functions
│   ├── train_naive.py      # Baseline MLP training loop
│   ├── train_pinn.py       # PINN multi-objective training loop
│   ├── train_world_model.py# PIWM training loop (Gradient Clipping, Dopri5)
│   └── utils.py            # Evaluation and visualization tools
├── tests/                  # Unit tests for physical consistency
│   └── test_dynamics.py
├── requirements.txt        # Python dependencies
├── requirement.txt         # Alternate dependency list
├── README.md               # Project documentation
└── report.pdf              # Full academic research paper
```

## 🚀 Getting Started

### 1. Environment Setup
Clone the repository and install the dependencies within a virtual environment:

Bash

```bash
git clone https://github.com/Optimus2007/physics-informed-world-models.git
cd physics-informed-world-models
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Pipeline
Execute the modules in the following order to replicate the research findings:

**Step A: Generate Ground-Truth Data**

Bash

```bash
python data/generate_orbit.py
```

**Step B: Train the Naive Baseline**

Bash

```bash
python src/train_naive.py
```

**Step C: Train the PINN**

Bash

```bash
python src/train_pinn.py
```

**Step D: Train the Physics-Informed World Model**

Bash

```bash
python src/train_world_model.py
```

### 3. Run Unit Tests
To verify the mathematical integrity of the underlying physics engine:

Bash

```bash
python -m unittest tests/test_dynamics.py
```

## 🔮 Future Work: Model-Based Reinforcement Learning (MBRL)
The ability of this architecture to cleanly map raw physical states into a causal, physics-bound latent representation establishes a robust foundation for Reinforcement Learning.

Future iterations of this repository will focus on embedding this PIWM as the "imagination engine" for RL agents (analogous to the Dreamer architecture). By enforcing strict thermodynamic priors during the RL latent planning phase, I aim to develop highly sample-efficient and provably safe control policies for complex physical environments.

## 📄 License
This project uses the MIT license terms for research and educational use.

*Developed by Aditya Raj.*
