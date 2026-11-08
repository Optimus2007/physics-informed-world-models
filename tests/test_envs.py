"""GATE A tests."""
import math
import numpy as np
import torch
import pytest
from src.control.envs import Pendulum, CartPole, Acrobot
from src.control.envs.pendulum import pendulum_deriv_np
from src.control.envs.cartpole import cartpole_deriv_np
from src.control.envs.acrobot import acrobot_deriv_np
from src.control.envs.wrappers import make_gym_pendulum, gym_pendulum_accel

# RK4 with dt=0.05 -> relative energy drift tolerance 1e-4 over 1000 steps (documented in DECISIONS D2)
TOL = 1e-4


def _drift(env, x0, n=1000):
    x = torch.tensor([x0], dtype=torch.float64)
    u = torch.zeros(1, 1, dtype=torch.float64)
    e0 = env.energy(x)
    emax = 0.0
    for _ in range(n):
        x = env.step(x, u)
        emax = max(emax, abs(float(env.energy(x) - e0)))
    scale = max(1.0, float(abs(e0)))
    return emax / scale


def test_energy_pendulum():
    assert _drift(Pendulum(), [2.0, 0.5]) < TOL


def test_energy_cartpole():
    assert _drift(CartPole(), [0.0, 2.0, 0.3, 0.5]) < TOL


def test_energy_acrobot():
    assert _drift(Acrobot(), [1.0, 0.7, 0.2, -0.3]) < TOL


def test_energy_decays_with_friction():
    env = Pendulum(b=0.2)
    x = torch.tensor([[2.0, 0.5]], dtype=torch.float64)
    e0 = env.energy(x)
    for _ in range(200):
        x = env.step(x, torch.zeros(1, 1, dtype=torch.float64))
    assert env.energy(x) < e0


def test_torch_numpy_agree():
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.uniform(-2, 2, 2); u = rng.uniform(-2, 2)
        d_t = Pendulum().deriv(torch.tensor(s)[None], torch.tensor([[u]]))[0].numpy()
        np.testing.assert_allclose(d_t, pendulum_deriv_np(s, u, 1.0, 1.0, 10.0, 0.0), atol=1e-12)
        s = rng.uniform(-2, 2, 4)
        e = CartPole(b=0.1)
        d_t = e.deriv(torch.tensor(s)[None], torch.tensor([[u]]))[0].numpy()
        p = e.params
        np.testing.assert_allclose(d_t, cartpole_deriv_np(s, u, p["M"], p["m"], p["l"], p["g"], p["b"]), atol=1e-12)


def test_numpy_rk4_trajectory_matches_torch():
    from src.dynamics import rk4_step  # the repo's existing NumPy RK4
    env = Pendulum()
    s = np.array([1.0, 0.0]); xt = torch.tensor([s])
    for _ in range(200):
        s = rk4_step(s, 0.0, env.dt, lambda st, t: pendulum_deriv_np(st, 0.0, 1.0, 1.0, 10.0, 0.0))
        xt = env.step(xt, torch.zeros(1, 1, dtype=torch.float64))
    np.testing.assert_allclose(xt[0].numpy(), s, atol=1e-9)


def test_pendulum_matches_gymnasium_dynamics():
    for (g, m, l) in [(10.0, 1.0, 1.0), (5.0, 2.0, 0.7)]:
        genv = make_gym_pendulum(g, m, l)
        sim = Pendulum(g=g, m=m, l=l, umax=2.0)
        for th, w, u in [(0.5, 0.0, 1.0), (-2.0, 1.0, -1.5), (3.0, -2.0, 0.3)]:
            a_gym = gym_pendulum_accel(genv, th, w, u)
            a_sim = sim.deriv(torch.tensor([[th, w]]), torch.tensor([[u]]))[0, 1].item()
            assert a_gym == pytest.approx(a_sim, rel=1e-4, abs=1e-4)  # gym is float32


@pytest.mark.parametrize("g,l", [(10.0, 1.0), (5.0, 1.0), (20.0, 1.0), (10.0, 0.5)])
def test_small_oscillation_period(g, l):
    """Hanging pendulum (theta=pi): w^2 = 3g/(2l) for the rod => T = 2*pi*sqrt(2l/(3g))."""
    env = Pendulum(g=g, l=l, umax=2.0)
    x = torch.tensor([[math.pi + 0.01, 0.0]], dtype=torch.float64)
    zero = torch.zeros(1, 1, dtype=torch.float64)
    ts, prev = [], x[0, 1].item(); t = 0.0; env.dt = 0.002
    xs = []
    for i in range(int(6 / env.dt)):
        x = env.step(x, zero); t += env.dt
        xs.append((t, x[0, 0].item() - math.pi))
    # upward zero crossings of displacement
    cross = [xs[i][0] - xs[i][1] * (xs[i + 1][0] - xs[i][0]) / (xs[i + 1][1] - xs[i][1])
             for i in range(len(xs) - 1) if xs[i][1] < 0 <= xs[i + 1][1]]
    period = np.mean(np.diff(cross))
    assert period == pytest.approx(2 * math.pi * math.sqrt(2 * l / (3 * g)), rel=1e-3)
    # theory ratio: period scales as sqrt(l/g)


def test_param_shift_changes_dynamics():
    a = Pendulum().deriv(torch.tensor([[1.0, 0.0]]), torch.zeros(1, 1))
    b = Pendulum().set_params(g=5.0).deriv(torch.tensor([[1.0, 0.0]]), torch.zeros(1, 1))
    assert abs(b[0, 1]) == pytest.approx(abs(a[0, 1]) / 2, rel=1e-9)
    with pytest.raises(KeyError):
        Pendulum().set_params(foo=1)
