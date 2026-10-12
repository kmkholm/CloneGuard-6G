"""Build CSI replay files for the testbed gNBs from the held-out DICHASUS day-2 recording (cf12, never used
for training/threshold).  The recording is cut into trajectory continuous segments of 600 snapshots (150 s replayed at 4/s);
each gNB receives a disjoint set of segments.  Output: emulation/traj_gnb<k>.npz with Hr/Hi (float16),
tid, pos.  python make_traj.py 4"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_dichasus
n_gnb = int(sys.argv[1]) if len(sys.argv) > 1 else 4
D = load_dichasus("cf12")
from data import _segment
D["tid"] = _segment(D["t"], 600, pos=D["pos"])      # continuous stretches of 600 snapshots (150 s at 4/s)
H, pos, tid = D["H"], D["pos"], D["tid"]
segs = np.array([u for u in np.unique(tid) if (tid == u).sum() >= 600])
rng = np.random.default_rng(0); rng.shuffle(segs)
out = os.path.join(os.path.dirname(__file__), "..", "emulation")
for k in range(n_gnb):
    s = segs[k::n_gnb]
    m = np.isin(tid, s)
    np.savez_compressed(os.path.join(out, f"traj_gnb{k+1}.npz"), Hr=H[m].real.astype(np.float16), Hi=H[m].imag.astype(np.float16),
                        tid=tid[m], pos=pos[m])
    print(f"gnb{k+1}: {m.sum()} snapshots, {len(s)} segments", sorted(s)[:8])
