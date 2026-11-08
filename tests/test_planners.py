import math, torch
from src.control.envs import Pendulum, CartPole
from src.control.planners.mppi import MPPI, CEM, run_episode


def test_mppi_pendulum_swingup_true_model():
    env = Pendulum()
    pl = MPPI(env.step, env.reward, horizon=40, n_samples=500, lam=0.5, sigma=1.0, umax=2.0, seed=0)
    X, U, R = run_episode(env, pl, torch.tensor([math.pi, 0.0], dtype=torch.float64), 200)
    th = (X[-50:, 0] + math.pi) % (2 * math.pi) - math.pi
    assert th.abs().max() < 0.3
    assert U.abs().max() <= 2.0 + 1e-9


def test_mppi_cartpole_swingup_true_model():
    env = CartPole()
    pl = MPPI(env.step, env.reward, horizon=50, n_samples=1000, lam=0.5, sigma=4.0, umax=10.0, seed=1)
    X, U, R = run_episode(env, pl, env.sample_init(1, torch.Generator().manual_seed(1))[0], 300)
    th = (X[-50:, 1] + math.pi) % (2 * math.pi) - math.pi
    assert th.abs().max() < 0.3


def test_mppi_deterministic_given_seed():
    env = Pendulum(); x0 = torch.tensor([3.0, 0.0], dtype=torch.float64)
    mk = lambda: MPPI(env.step, env.reward, horizon=10, n_samples=50, seed=7)
    assert torch.equal(mk().plan(x0), mk().plan(x0))
