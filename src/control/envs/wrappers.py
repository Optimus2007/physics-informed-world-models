"""Gymnasium bridge. Gymnasium Pendulum-v1 exposes g, m, l via env.unwrapped; we use it as an
independent reference for our pendulum dynamics (see tests/test_envs.py)."""
import gymnasium as gym
import numpy as np


def make_gym_pendulum(g=10.0, m=1.0, l=1.0):
    env = gym.make("Pendulum-v1")
    u = env.unwrapped
    u.g, u.m, u.l = g, m, l
    return env


def gym_pendulum_accel(env, theta, omega, torque):
    """One Gymnasium step from (theta, omega); returns the acceleration it implies."""
    env.reset(seed=0)
    env.unwrapped.state = np.array([theta, omega])
    env.step(np.array([torque], dtype=np.float32))
    new_omega = env.unwrapped.state[1]
    return (new_omega - omega) / env.unwrapped.dt
