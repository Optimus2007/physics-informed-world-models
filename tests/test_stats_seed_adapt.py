import numpy as np, torch, pytest
from scipy import stats as sps
from src.control.eval.stats import bootstrap_ci, iqm
from src.control.envs import Pendulum
from src.control.models.physics_structured import PhysicsStructured
from src.control.adapt.sysid import adapt_physics, collect_adaptation_data


def test_bootstrap_ci_known_sample():
    x = np.arange(1, 101, dtype=float)            # mean 50.5, sd 29.0 -> SE ~ 2.9, 95% CI ~ +-5.7
    m, lo, hi = bootstrap_ci(x, n_boot=4000)
    assert m == 50.5 and lo < 50.5 < hi
    assert (hi - lo) == pytest.approx(2 * 1.96 * 29.01 / 10, rel=0.1)


def test_bootstrap_ci_coverage():
    rng = np.random.default_rng(0); hit = 0
    for i in range(200):
        s = rng.normal(0, 1, 20); _, lo, hi = bootstrap_ci(s, n_boot=500, seed=i); hit += lo < 0 < hi
    assert 0.88 <= hit / 200 <= 0.99               # nominal 95%; percentile bootstrap slightly liberal at n=20


def test_iqm():
    x = [0, 1, 2, 3, 4, 5, 6, 7, 1000, 2000, 3, 4]
    assert iqm(x) == pytest.approx(sps.trim_mean(x, 0.25)) and abs(iqm(x) - np.mean(x)) > 1


def test_seeding_reproducible():
    from src.control.eval.data import collect_random
    a = collect_random(Pendulum(), 3, 5, seed=4); b = collect_random(Pendulum(), 3, 5, seed=4)
    assert torch.equal(a.x, b.x) and torch.equal(a.xn, b.xn)
    c = collect_random(Pendulum(), 3, 5, seed=5)
    assert not torch.equal(a.x, c.x)


def test_sysid_recovers_gravity_from_50_transitions():
    """Synthetic: model trained at g=10, real env has g=5; 50 transitions; adapt g only (known geometry)."""
    shifted = Pendulum(g=5.0)
    data = collect_adaptation_data(shifted, 50, 5, seed=0)
    m = PhysicsStructured(Pendulum(), fixed=("m", "l", "b")); m.set_norm(data.x, data.xn)
    m2 = adapt_physics(m, data, steps=300, lr=3e-2)
    assert m2.param_dict()["g"] == pytest.approx(5.0, rel=0.02)


def test_sysid_all_params_matches_dynamics():
    """Adapting all (m,l,g): individual values non-identifiable, but identifiable combos must match."""
    shifted = Pendulum(g=5.0, l=1.3)
    data = collect_adaptation_data(shifted, 50, 5, seed=1)
    m = PhysicsStructured(Pendulum(), fixed=("b",)); m.set_norm(data.x, data.xn)
    p = adapt_physics(m, data, steps=800, lr=3e-2).param_dict()
    assert 1.5 * p["g"] / p["l"] == pytest.approx(1.5 * 5.0 / 1.3, rel=0.05)
    assert 3 / (p["m"] * p["l"] ** 2) == pytest.approx(3 / 1.3 ** 2, rel=0.05)
