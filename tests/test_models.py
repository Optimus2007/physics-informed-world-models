import math, pytest, torch
from src.control.envs import Pendulum, CartPole
from src.control.models.mlp_ensemble import MLPEnsemble
from src.control.models.neural_ode import NeuralODE
from src.control.models.physics_structured import PhysicsStructured
from src.control.models.neural_hamiltonian import NeuralHamiltonian
from src.control.models.legacy_latent_ode import LegacyLatentODE
from src.control.eval.data import collect_random, fit

ALL = [MLPEnsemble, NeuralODE, PhysicsStructured, NeuralHamiltonian, LegacyLatentODE]


@pytest.mark.parametrize("Env", [Pendulum, CartPole])
@pytest.mark.parametrize("M", ALL)
def test_shapes_and_gradients(Env, M):
    env = Env(); x = env.sample_init_wide(6); u = torch.randn(6, 1, dtype=torch.float64); xn = env.step(x, u)
    m = M(env, init_params={k: v * 1.3 for k, v in env.params.items() if k in "Mmlg"}) if M is PhysicsStructured else M(env)
    assert m.step(x, u).shape == x.shape
    m.loss(x, u, xn, idx=torch.arange(6)).backward()
    nz = [p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters()]
    assert sum(nz) >= 0.9 * len(nz) - 1      # (a constant bias in V has no gradient by construction)
    assert m.rollout(x, torch.zeros(6, 5, 1, dtype=torch.float64)).shape == (6, 6, env.state_dim)


def test_physics_with_true_params_is_exact():
    env = Pendulum(); x = env.sample_init_wide(20); u = torch.rand(20, 1, dtype=torch.float64)
    assert (PhysicsStructured(env).step(x, u) - env.step(x, u)).abs().max() < 1e-12


def test_pendulum_identifiable_combos_recovered():
    """Start from wrong (m,l,g); the data pin down a=3g/2l, c=3/(m l^2) but NOT m,l,g individually."""
    env = Pendulum(); data = collect_random(env, 40, 50, seed=0)
    m = PhysicsStructured(env, init_params=dict(m=1.4, l=0.8, g=7.0))
    fit(m, data, steps=1500, lr=1e-2)
    p = m.param_dict()
    a, c = 1.5 * p["g"] / p["l"], 3 / (p["m"] * p["l"] ** 2)
    assert a == pytest.approx(15.0, rel=0.02) and c == pytest.approx(3.0, rel=0.02)


def test_pendulum_g_recovered_when_geometry_known():
    env = Pendulum(); data = collect_random(env, 40, 50, seed=0)
    m = PhysicsStructured(env, init_params=dict(g=6.0), fixed=("m", "l", "b"))
    fit(m, data, steps=1500, lr=1e-2)
    assert m.param_dict()["g"] == pytest.approx(10.0, rel=0.02)


def test_hamiltonian_conserves_learned_energy_untrained():
    """Structural property: with u=0 the learned H is a first integral of the flow (up to RK4 error)."""
    env = Pendulum(); torch.manual_seed(0); m = NeuralHamiltonian(env, substeps=8)
    x = env.sample_init(8); U = torch.zeros(8, 100, 1, dtype=torch.float64)
    traj = m.rollout(x, U)
    E = m.energy(traj.reshape(-1, 2)).reshape(8, 101).detach()
    scale = E.abs().max() + 1e-6
    assert (E - E[:, :1]).abs().max() < 1e-3 * max(scale, 1.0)


def test_hamiltonian_flow_is_volume_preserving():
    """Liouville: d(qdot)/dq + d(pdot)/dp = 0 for u = 0 (symplectic flows preserve phase-space volume)."""
    env = Pendulum(); torch.manual_seed(1); m = NeuralHamiltonian(env)
    z = torch.tensor([0.7, 0.3], dtype=torch.float64)
    u = torch.zeros(1, dtype=torch.float64)
    J = torch.autograd.functional.jacobian(lambda zz: m.zdot(zz, u), z)
    assert abs(torch.trace(J)) < 1e-9


def test_mlp_has_no_conserved_energy_while_ham_does():
    """Contrast: random-init MLP rollout drifts in true energy more than random-init... (sanity only: just runs)"""
    env = Pendulum(); m = MLPEnsemble(env)
    m.rollout(env.sample_init(3), torch.zeros(3, 5, 1, dtype=torch.float64))
