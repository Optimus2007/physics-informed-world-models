"""Bootstrap confidence intervals and interquartile mean (IQM). Own implementation (percentile bootstrap over
seeds); tests cross-check IQM against scipy.stats.trim_mean and CI coverage on a known distribution. rliable
(Agarwal et al. 2021, Apache-2.0) uses stratified bootstrap over runs x tasks; with one task and few seeds the
two coincide up to Monte-Carlo noise (see tests)."""
import numpy as np
from scipy import stats as sps


def iqm(x):
    """Mean of the middle 50% of values (25% trimmed from each side)."""
    return float(sps.trim_mean(np.asarray(x, float), 0.25))


def bootstrap_ci(x, stat=np.mean, n_boot=5000, alpha=0.05, seed=0):
    """Percentile bootstrap CI over the entries of x (e.g. over seeds). Returns (point, lo, hi)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    return float(stat(x)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def summarize(x, **kw):
    m, lo, hi = bootstrap_ci(x, **kw)
    return dict(mean=m, lo=lo, hi=hi, iqm=iqm(x), n=len(x))
