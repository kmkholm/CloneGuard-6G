"""Datasets and episode sampler for clone-resilient physical-layer authentication (PLA).

Every dataset is a set of mobile trajectories with cell-free uplink CSI H (N, A, M, F) complex,
positions pos (N, 2), trajectory id tid (N,), time t (N,).
Episode = (history of K legitimate snapshots, query snapshot, label). Label 0 = legitimate continuation,
label 1 = clone/spoofer transmitting the same identity from another position (distance d to the true UE).
"""
import os, json
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DIST_BINS = [0.25, 0.5, 1, 2, 5, 10, 20, 50, 1e9]


def _segment(t, seg_len, gap=0.3, pos=None, max_step=0.08):
    """split a recording into continuous pseudo-trajectories: a new segment starts at every time gap > `gap` s,
    at every position jump > `max_step` m (a third of a wavelength at 1.27 GHz; the sounder occasionally skips
    records, and a jump of a wavelength is indistinguishable from another transmitter) and every seg_len samples"""
    tid = np.zeros(len(t), int); k = 0; n = 0
    for i in range(1, len(t)):
        n += 1
        jump = pos is not None and np.linalg.norm(pos[i] - pos[i - 1]) > max_step
        if t[i] - t[i - 1] > gap or jump or n >= seg_len:
            k += 1; n = 0
        tid[i] = k
    return tid


def load_dichasus(name):
    """DICHASUS cell-free (real): 4 arrays x 8 slots, 24 antennas common to both days (others zero)."""
    d = np.load(os.path.join(ROOT, "dichasus", f"{name}.npz"))
    mp = json.load(open(os.path.join(ROOT, "dichasus", "common24.json")))
    idx = mp["cf0x_idx"] if name.startswith("cf0") else mp["cf1x_idx"]
    H = np.zeros((len(d["H"]), 4, 8, d["H"].shape[-1]), np.complex64)
    arr = {"A": 0, "B": 1, "C": 2, "D": 3}
    for k, i in enumerate(idx):
        H[:, arr[mp["array"][k]], mp["row"][k] * 4 + mp["col"][k]] = d["H"][:, i]
    t = d["t"]; tid = _segment(t, 400, pos=d["pos"][:, :2])
    return dict(H=H, pos=d["pos"][:, :2], tid=tid, t=t, veh=np.zeros(len(t), bool), noise=None,
                name=f"dichasus-{name}")


def load_rt(name, ptx_dbm=23.0, nf_db=7.0, proc_gain_db=12.0):
    d = np.load(os.path.join(ROOT, "rt", f"{name}.npz"))
    bw = float(d["bw"])
    # noise variance relative to a unit-power transmit signal (thermal noise, NF, SRS processing gain)
    n0 = 10 ** ((-174 + 10 * np.log10(bw) + nf_db - proc_gain_db - ptx_dbm) / 10)
    return dict(H=d["H"], pos=d["pos"], tid=d["tid"], t=d["t"], veh=d["veh"], noise=n0, name=name,
                ap=d["ap"])


def split_by_traj(D, fr=(0.6, 0.2, 0.2), seed=0):
    u = np.unique(D["tid"]); rng = np.random.default_rng(seed); rng.shuffle(u)
    a, b = int(fr[0] * len(u)), int((fr[0] + fr[1]) * len(u))
    return [np.isin(D["tid"], s) for s in (u[:a], u[a:b], u[b:])]


def subset(D, m):
    return {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == m.shape else v) for k, v in D.items()}


class EpisodeSampler:
    """GPU-resident sampler. Indices of valid anchors: positions with >= K*gap predecessors in the same trajectory."""

    def __init__(self, D, K, device, gaps=(1,), max_gap=16):
        self.K, self.dev, self.gaps = K, device, gaps
        H = D["H"]
        self.H = torch.from_numpy(np.stack([H.real, H.imag], -1).astype(np.float16)).to(device)  # N,A,M,F,2
        self.noise = D["noise"]
        self.pos = torch.from_numpy(D["pos"]).float().to(device)
        tid = D["tid"]; N = len(tid)
        # first index of each sample's trajectory
        start = np.zeros(N, int)
        for i in range(1, N):
            start[i] = start[i - 1] if tid[i] == tid[i - 1] else i
        self.start = torch.from_numpy(start).to(device)
        self.anchor = torch.from_numpy(np.nonzero(np.arange(N) - start >= K)[0]).to(device)   # gap >= 1 feasible
        self.N = N

    def _spoof_index(self, a, dmin, dmax, g):
        """for each anchor pick a random snapshot whose position is in [dmin, dmax) from the anchor (rejection)"""
        B = len(a); out = torch.full((B,), -1, device=self.dev, dtype=torch.long)
        todo = torch.arange(B, device=self.dev)
        for _ in range(64):
            if len(todo) == 0: break
            cand = torch.randint(0, self.N, (len(todo), 32), generator=g, device=self.dev)
            d = (self.pos[cand] - self.pos[a[todo]][:, None]).norm(dim=-1)
            ok = (d >= dmin[todo][:, None]) & (d < dmax[todo][:, None])
            has = ok.any(1)
            first = ok.float().argmax(1)
            out[todo[has]] = cand[has, first[has]]
            todo = todo[~has]
        return out

    def sample(self, B, g, dist_lo=0.25, dist_hi=1e9, gap=None, p_spoof=0.5, anchors=None):
        dev = self.dev
        a = anchors if anchors is not None else self.anchor[torch.randint(0, len(self.anchor), (B,), generator=g, device=dev)]
        B = len(a)
        if gap is None:
            gs = torch.tensor(self.gaps, device=dev)[torch.randint(0, len(self.gaps), (B,), generator=g, device=dev)]
        else:
            gs = torch.full((B,), gap, device=dev)
        gs = torch.minimum(gs, torch.clamp((a - self.start[a]) // self.K, min=1))             # stay inside the trajectory
        hist = a[:, None] - gs[:, None] * torch.arange(self.K, 0, -1, device=dev)[None]      # B,K
        y = (torch.rand(B, generator=g, device=dev) < p_spoof).long()
        # log-uniform spoofer distance
        lo, hi = np.log(dist_lo), np.log(min(dist_hi, 400.0))
        dd = torch.exp(lo + (hi - lo) * torch.rand(B, generator=g, device=dev))
        dmin, dmax = dd / 1.25, dd * 1.25
        sp = self._spoof_index(a, dmin, dmax, g)
        y = torch.where(sp < 0, torch.zeros_like(y), y)
        q = torch.where(y == 1, sp, a)
        dist = (self.pos[q] - self.pos[a]).norm(dim=-1)
        X = self.H[torch.cat([hist, q[:, None]], 1)].float()                     # B,K+1,A,M,F,2
        return X, y, dist, a

    def add_noise(self, X, g, snr_offset_db=0.0):
        if self.noise is None: return X
        s = np.sqrt(self.noise * 10 ** (-snr_offset_db / 10) / 2)
        return X + s * torch.randn(X.shape, generator=g, device=X.device)


def normalise(X, eps=1e-12):
    """X: B,T,A,M,F,2 -> unit-norm snapshot, per-AP common phase removed (strongest element real),
    plus per-AP power share (dB).  Returns complex-as-2ch tensor and log power B,T,A."""
    Hc = torch.complex(X[..., 0], X[..., 1])                                     # B,T,A,M,F
    p_ap = (Hc.abs() ** 2).sum((-1, -2))                                          # B,T,A
    tot = p_ap.sum(-1, keepdim=True) + eps
    Hc = Hc / tot.sqrt()[..., None, None]
    flat = Hc.flatten(-2)
    k = flat.abs().argmax(-1, keepdim=True)
    ref = torch.gather(flat, -1, k)
    rot = torch.where(ref.abs() > 0, ref.conj() / (ref.abs() + eps), torch.ones_like(ref))
    Hc = Hc * rot[..., None]
    lp = 10 * torch.log10(p_ap / tot + 1e-6)                                      # <= 0 dB, floor -60
    return torch.stack([Hc.real, Hc.imag], -1), lp
