"""Neural Hamiltonian with control, SymODEN-style (Zhong, Dey, Chakraborty, ICLR 2020; arXiv:1909.12077).
Re-implemented in this repo's own style (not copied). For d degrees of freedom, q in R^d, p in R^d:
    H(q,p) = 1/2 p^T Minv(q) p + V(q)          Minv(q) = L L^T + eps I, L lower-triangular from an MLP
    dq/dt = dH/dp,   dp/dt = -dH/dq + g(q) u   (g(q): d x m, MLP)       no parameters of the physics are given.
Data are in (q, qdot); the model converts to momenta with its OWN learned mass matrix p = Minv(q)^{-1} qdot,
integrates Hamilton's equations with RK4, then converts back. No dissipation term (so friction is out of its
hypothesis class: that is the point of Phase F). Angles enter nets as (cos, sin)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import WorldModel, mlp, rk4, DT


class NeuralHamiltonian(WorldModel):
    name = "neural_hamiltonian"

    def __init__(self, env, hidden=(128, 128), substeps=2):
        super().__init__(env)
        self.d = self.n // 2
        self.sub = substeps
        self.nq_feat = self.nfeat - self.d          # features of q only (q = first d entries, as in all envs)
        self.V = mlp(self.nq_feat, 1, hidden)
        self.Lnet = mlp(self.nq_feat, self.d * (self.d + 1) // 2, hidden)
        self.G = mlp(self.nq_feat, self.d * self.m, hidden)
        self.register_buffer("gscale", torch.ones(1, dtype=DT))

    def qfeat(self, q):
        cols = []
        for i in range(self.d):
            cols += [torch.cos(q[..., i:i + 1]), torch.sin(q[..., i:i + 1])] if i in self.angle_idx else [q[..., i:i + 1]]
        return torch.cat(cols, -1)

    def Minv(self, q):
        out = self.Lnet(self.qfeat(q))
        L = torch.zeros(*q.shape[:-1], self.d, self.d, dtype=DT)
        L = self._fill(L, out)
        diag = F.softplus(torch.diagonal(L, dim1=-2, dim2=-1)) + 1e-3
        L = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) + torch.diag_embed(diag)
        return L @ L.transpose(-1, -2) + 1e-4 * torch.eye(self.d, dtype=DT)

    def _fill(self, L, vals):
        rows = []
        k = 0
        for i in range(self.d):
            row = []
            for j in range(self.d):
                if j <= i:
                    row.append(vals[..., k]); k += 1
                else:
                    row.append(torch.zeros_like(vals[..., 0]))
            rows.append(torch.stack(row, -1))
        return torch.stack(rows, -2)

    def Gmat(self, q):
        return self.G(self.qfeat(q)).reshape(*q.shape[:-1], self.d, self.m)

    def hamiltonian(self, q, p):
        Mi = self.Minv(q)
        kin = 0.5 * torch.einsum("...i,...ij,...j->...", p, Mi, p)
        return kin + self.V(self.qfeat(q))[..., 0]

    def zdot(self, z, u):
        d = self.d
        with torch.enable_grad():
            q = z[..., :d]
            q = q if q.requires_grad else q.detach().requires_grad_(True)
            p = z[..., d:]
            p = p if p.requires_grad else p.detach().requires_grad_(True)
            H = self.hamiltonian(q, p)
            dHdq, dHdp = torch.autograd.grad(H.sum(), (q, p), create_graph=True)
        gu = torch.einsum("...ij,...j->...i", self.Gmat(q), u / self.umax) * self.gscale
        out = torch.cat([dHdp, -dHdq + gu], -1)
        return out if torch.is_grad_enabled() else out.detach()

    def to_z(self, x):
        q, qd = x[..., :self.d], x[..., self.d:]
        p = torch.linalg.solve(self.Minv(q), qd[..., None])[..., 0]
        return torch.cat([q, p], -1)

    def from_z(self, z):
        q, p = z[..., :self.d], z[..., self.d:]
        return torch.cat([q, torch.einsum("...ij,...j->...i", self.Minv(q), p)], -1)

    def step(self, x, u):
        z = rk4(self.zdot, self.to_z(x), u, self.dt, self.sub)
        return self.from_z(z)

    def energy(self, x):
        z = self.to_z(x)
        return self.hamiltonian(z[..., :self.d], z[..., self.d:])
