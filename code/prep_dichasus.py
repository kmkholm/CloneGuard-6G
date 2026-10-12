"""Convert DICHASUS cell-free tfrecords to compact npz.

Per record:
  1. sampling-time-offset correction: coarse delay from the 4x-oversampled power-delay profile summed over
     antennas, then fine residual from the antenna-summed phase slope across subcarriers (Day 1 drifts by
     ~9.5 taps per record);
  2. 1024 subcarriers -> 64 sub-bands (mean of 16 adjacent subcarriers; delay-domain low-pass).
Day-1 files (cf0x) interleave two transmit streams (alternate records are 0.99-correlated with each other
but only ~0.2 with their neighbours, consistent with two alternating transmit antennas). Records are split
greedily into streams by correlation with each stream's last snapshot; only the larger stream is kept.
usage: python prep_dichasus.py cf02 [cf03 ...]
"""
import os, sys, time
import numpy as np
from dichasus_io import iterate

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "dichasus")
NB = 64


def sto_fix(h, os_=4):
    N = h.shape[-1]
    P = (np.abs(np.fft.ifft(h, n=N * os_, axis=-1)) ** 2).sum(0)
    tau = np.argmax(P) / os_
    if tau > N / 2: tau -= N
    h = h * np.exp(2j * np.pi * np.arange(N) * tau / N)
    s = (h[:, 1:] * h[:, :-1].conj()).sum()
    return h * np.exp(-1j * np.angle(s) * np.arange(N))


def corr(a, b):
    return np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


for name in sys.argv[1:]:
    src = os.path.join(ROOT, f"dichasus-{name}.tfrecords")
    dst = os.path.join(ROOT, f"{name}.npz")
    t0 = time.time(); H, P, T, S = [], [], [], []
    for r in iterate(src):
        c = r["csi"]
        h = sto_fix((c[..., 0] + 1j * c[..., 1]).astype(np.complex128))
        h = h.reshape(h.shape[0], NB, -1).mean(-1).astype(np.complex64)
        H.append(h); P.append(r["pos-tachy"]); T.append(r["time"][0]); S.append(r["snr"])
    o = np.argsort(T)
    H = np.stack(H)[o]; P = np.stack(P).astype(np.float32)[o]; T = np.asarray(T)[o]; S = np.stack(S)[o]
    stream = np.zeros(len(H), int)
    if name.startswith("cf0"):
        last = [0, None]
        for i in range(1, len(H)):
            if last[1] is None:
                stream[i] = 1; last[1] = i; continue
            c0, c1 = corr(H[i].ravel(), H[last[0]].ravel()), corr(H[i].ravel(), H[last[1]].ravel())
            k = 0 if c0 >= c1 else 1
            stream[i] = k; last[k] = i
    keep = stream == np.bincount(stream).argmax()
    lag1 = np.mean([corr(H[keep][i].ravel(), H[keep][i + 1].ravel()) for i in range(0, keep.sum() - 1, 50)])
    np.savez(dst, H=H[keep], pos=P[keep], t=T[keep], snr=S[keep])
    print(name, "records", len(H), "stream sizes", np.bincount(stream), "kept", int(keep.sum()),
          f"{time.time()-t0:.0f}s", "dt med", np.median(np.diff(T[keep])).round(3), "lag1 corr", round(lag1, 3))
