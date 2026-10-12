"""Fast street-graph trajectories (scipy sparse Dijkstra) for gen_rt.py."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


class StreetGraph:
    def __init__(self, m, xs, ys):
        self.m, self.xs, self.ys = m, xs, ys
        idx = -np.ones(m.shape, int); cells = np.argwhere(m); idx[m] = np.arange(len(cells))
        self.cells = cells
        rows, cols, w = [], [], []
        for da, db, ww in ((1, 0, 1.0), (0, 1, 1.0), (1, 1, 1.414), (1, -1, 1.414)):
            a = cells; b = a + (da, db)
            ok = (b[:, 0] >= 0) & (b[:, 0] < m.shape[0]) & (b[:, 1] >= 0) & (b[:, 1] < m.shape[1])
            ok[ok] &= m[b[ok, 0], b[ok, 1]]
            ia, ib = idx[a[ok, 0], a[ok, 1]], idx[b[ok, 0], b[ok, 1]]
            rows += [ia, ib]; cols += [ib, ia]; w += [np.full(len(ia), ww)] * 2
        n = len(cells)
        self.G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()

    def path(self, s, g):
        d, pred = dijkstra(self.G, indices=s, return_predecessors=True, limit=600)
        if not np.isfinite(d[g]):
            return None
        p = [g]
        while p[-1] != s:
            p.append(pred[p[-1]])
        c = self.cells[p[::-1]]
        return np.c_[self.xs[c[:, 0]], self.ys[c[:, 1]]].astype(float)


def trajectory(sg, rng, step, n_samples):
    n = len(sg.cells)
    for _ in range(50):
        a, b = rng.integers(n), rng.integers(n)
        if np.hypot(*(sg.cells[a] - sg.cells[b])) < 40:
            continue
        P = sg.path(a, b)
        if P is None or len(P) < 20:
            continue
        k = 5; Ps = np.copy(P)
        for d in range(2):
            Ps[:, d] = np.convolve(np.pad(P[:, d], k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Ps, axis=0), axis=1))]
        L = min(seg[-1], step * (n_samples - 1))
        s = np.linspace(0, L, int(L / step) + 1)
        if len(s) < n_samples // 2:
            continue
        return np.c_[np.interp(s, seg, Ps[:, 0]), np.interp(s, seg, Ps[:, 1])]
    return None
