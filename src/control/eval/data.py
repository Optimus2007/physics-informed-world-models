"""Transition datasets, random-action collection, and the common model trainer."""
import math, time
import torch

DT = torch.float64


class Data:
    """Flat transition store with episode ids (so train/val split is by episode, not by row)."""
    def __init__(self, x=None, u=None, xn=None, ep=None):
        z = lambda *s: torch.zeros(*s, dtype=DT)
        self.x, self.u, self.xn = x if x is not None else z(0, 1), u if u is not None else z(0, 1), xn if xn is not None else z(0, 1)
        self.ep = ep if ep is not None else torch.zeros(0, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def add(self, x, u, xn, ep_id):
        e = torch.full((len(x),), ep_id, dtype=torch.long)
        if len(self) == 0:
            self.x, self.u, self.xn, self.ep = x.clone(), u.clone(), xn.clone(), e
        else:
            self.x, self.u, self.xn, self.ep = (torch.cat([self.x, x]), torch.cat([self.u, u]),
                                                torch.cat([self.xn, xn]), torch.cat([self.ep, e]))

    def split(self, val_frac, gen):
        eps = self.ep.unique()
        perm = eps[torch.randperm(len(eps), generator=gen)]
        nval = max(1, int(round(val_frac * len(eps))))
        val_eps = perm[:nval]
        m = torch.isin(self.ep, val_eps)
        return (Data(self.x[~m], self.u[~m], self.xn[~m], self.ep[~m]), Data(self.x[m], self.u[m], self.xn[m], self.ep[m]))


def collect_random(env, n_episodes, ep_len=50, seed=0, wide=True, ep_offset=0):
    """Random uniform actions in [-umax,umax], fresh wide initial state per episode."""
    g = torch.Generator().manual_seed(seed)
    d = Data()
    x0 = env.sample_init_wide(n_episodes, g) if wide else env.sample_init(n_episodes, g)
    x = x0
    X, U, XN = [], [], []
    for t in range(ep_len):
        u = (torch.rand(n_episodes, env.action_dim, generator=g, dtype=DT) * 2 - 1) * env.params["umax"]
        xn = env.step(x, u)
        X.append(x); U.append(u); XN.append(xn); x = xn
    X, U, XN = (torch.stack(a, 1) for a in (X, U, XN))      # (E,T,*)
    for e in range(n_episodes):
        d.add(X[e], U[e], XN[e], ep_offset + e)
    return d


def fit(model, data, steps=3000, lr=1e-3, bs=256, val_frac=0.2, seed=0, patience=10, eval_every=100,
        params=None, verbose=False, renorm=True):
    """Train with Adam on the one-step normalised loss. Keeps best-validation weights.
    params: optional list of parameters to train (others frozen), e.g. only physical params for sysid."""
    g = torch.Generator().manual_seed(seed)
    tr, va = data.split(val_frac, g) if (val_frac > 0 and len(data.ep.unique()) > 1) else (data, data)
    if renorm:
        model.set_norm(tr.x, tr.xn)
    if hasattr(model, "fit_prepare"):
        model.fit_prepare(len(tr), g)
    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    best, best_state, bad, hist = 1e30, None, 0, []
    for s in range(1, steps + 1):
        idx = torch.randint(0, len(tr), (min(bs, len(tr)),), generator=g)
        model.train()
        loss = model.loss(tr.x[idx], tr.u[idx], tr.xn[idx], idx=idx)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
        if s % eval_every == 0 or s == steps:
            model.eval()
            with torch.no_grad() if not hasattr(model, "hamiltonian") else torch.enable_grad():
                v = float(((model.step(va.x, va.u).detach() - va.xn) / model.dstd).pow(2).mean())
            hist.append((s, float(loss.detach()), v))
            if verbose:
                print(s, round(float(loss), 5), round(v, 6), flush=True)
            if v < best - 1e-9:
                best, bad = v, 0; best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return dict(best_val=best, hist=hist)
