"""Generate cell-free uplink CSI along mobile user trajectories with Sionna RT (ray tracing).

Scenes / bands (6G):
  munich28 : Munich city scene, 28 GHz (FR2 mmWave), 100 MHz, 6 distributed APs (2x4 UPA, 6 m lamp-post height)
  sf7      : San Francisco scene, 7 GHz (FR3 upper mid-band), 200 MHz, 6 distributed APs
The UE (1 antenna) transmits an uplink SRS; by reciprocity it is the RT transmitter and the APs are receivers.
Outputs data/rt/<name>.npz with complex64 H (N, A, M, F) path-gain CFR (no noise), positions, trajectory ids,
time stamps, speed class, AP positions, street mask.  Noise is added later from a link budget.
usage: python gen_rt.py munich28 [n_traj]
"""
import os, sys, time, heapq, json
import numpy as np
import mitsuba as mi
mi.set_variant("cuda_ad_mono_polarized")
import drjit as dr
from traj_fast import StreetGraph, trajectory as traj_fast
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

CFG = {
    "munich28": dict(scene="munich", fc=28e9, bw=100e6, center=(0.0, 0.0), half=120.0, seed=1),
    "sf7": dict(scene="san_francisco", fc=7e9, bw=200e6, center=None, half=120.0, seed=2),
}
NAP, NF, AP_H, UE_H = 6, 32, 6.0, 1.5
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "rt")
os.makedirs(OUT, exist_ok=True)


def street_mask(scene, cx, cy, half, res=1.0):
    """walkable street cells = cells whose vertical ray hits the terrain mesh (not a building); the ground map G is
    the terrain height, filled by nearest street cell elsewhere (works for hilly scenes such as San Francisco)."""
    from scipy.ndimage import distance_transform_edt
    xs = np.arange(cx - half, cx + half + 1e-6, res); ys = np.arange(cy - half, cy + half + 1e-6, res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    o = mi.Point3f(mi.Float(X.ravel()), mi.Float(Y.ravel()), mi.Float(np.full(X.size, 500.0)))
    si = scene.mi_scene.ray_intersect(mi.Ray3f(o, mi.Vector3f(0, 0, -1)))
    z = 500.0 - np.array(si.t); valid = np.array(si.is_valid())
    shapes = scene.mi_scene.shapes()
    terr = [sh for sh in shapes if "terrain" in sh.id().lower() or "ground" in sh.id().lower()]
    if terr:
        hit_terr = np.zeros(X.size, bool)
        for sh in terr:
            hit_terr |= np.array(si.shape == mi.ShapePtr(sh))
        m = (hit_terr & valid).reshape(X.shape)
    else:                                   # flat scenes without a terrain object: lowest level
        gz = np.percentile(z[valid], 20); m = (np.abs(z - gz) < 0.6).reshape(X.shape) & valid.reshape(X.shape)
    Z = z.reshape(X.shape)
    idx = distance_transform_edt(~m, return_distances=False, return_indices=True)
    G = Z[idx[0], idx[1]]
    e = m.copy(); e[1:] &= m[:-1]; e[:-1] &= m[1:]; e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
    return xs, ys, e, G


def largest_component(m):
    lab = -np.ones(m.shape, int); best, bl = 0, -1; k = 0
    for s in zip(*np.nonzero(m)):
        if lab[s] >= 0: continue
        st = [s]; lab[s] = k; n = 0
        while st:
            a, b = st.pop(); n += 1
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c = (a + da, b + db)
                if 0 <= c[0] < m.shape[0] and 0 <= c[1] < m.shape[1] and m[c] and lab[c] < 0:
                    lab[c] = k; st.append(c)
        if n > best: best, bl = n, k
        k += 1
    return lab == bl


def shortest(m, s, g):
    D = {s: 0.0}; prev = {}; pq = [(0.0, s)]
    nb = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == g: break
        if d > D[u]: continue
        for da, db, w in nb:
            v = (u[0] + da, u[1] + db)
            if 0 <= v[0] < m.shape[0] and 0 <= v[1] < m.shape[1] and m[v]:
                nd = d + w
                if nd < D.get(v, 1e18):
                    D[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
    if g not in prev: return None
    p = [g]
    while p[-1] != s: p.append(prev[p[-1]])
    return p[::-1]


def trajectory(m, xs, ys, rng, step, n_samples):
    cells = np.argwhere(m)
    for _ in range(50):
        a, b = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
        if np.hypot(*(a - b)) < 40: continue
        p = shortest(m, tuple(a), tuple(b))
        if p is None or len(p) < 20: continue
        P = np.array([[xs[i], ys[j]] for i, j in p], float)
        # smooth (moving average) and resample at constant arc-length step
        k = 5; Ps = np.copy(P)
        for d in range(2): Ps[:, d] = np.convolve(np.pad(P[:, d], k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Ps, axis=0), axis=1))]
        L = min(seg[-1], step * (n_samples - 1))
        s = np.linspace(0, L, int(L / step) + 1)
        if len(s) < n_samples // 2: continue
        return np.c_[np.interp(s, seg, Ps[:, 0]), np.interp(s, seg, Ps[:, 1])]
    return None


def main(name, n_traj):
    c = CFG[name]; rng = np.random.default_rng(c["seed"])
    scene = load_scene(getattr(rt.scene, c["scene"]))
    scene.frequency = c["fc"]
    bb = scene.mi_scene.bbox()
    cx, cy = c["center"] if c["center"] else (0.5 * (bb.min[0] + bb.max[0]), 0.5 * (bb.min[1] + bb.max[1]))
    xs, ys, m, G = street_mask(scene, cx, cy, c["half"])
    m = largest_component(m)
    gz = float(np.median(G[m]))
    gnd = lambda x, y: float(G[int(round((x - xs[0]))), int(round((y - ys[0])))])
    print(name, "street cells", int(m.sum()), "of", m.size, "ground z median", gz, "range", G[m].min(), G[m].max())
    # APs: farthest-point sampling on street cells, 6 m above ground ("lamp-post" small cells)
    cells = np.argwhere(m); pts = np.c_[xs[cells[:, 0]], ys[cells[:, 1]]]
    pts = pts[(np.abs(pts[:, 0] - cx) < 0.7 * c["half"]) & (np.abs(pts[:, 1] - cy) < 0.7 * c["half"])]
    ap = [pts[rng.integers(len(pts))]]
    for _ in range(NAP - 1):
        d = np.min(np.linalg.norm(pts[:, None] - np.array(ap)[None], axis=2), axis=1); ap.append(pts[np.argmax(d)])
    ap = np.array(ap)
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=2, num_cols=4, vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="iso", polarization="V")
    for i, p in enumerate(ap):
        scene.add(Receiver(name=f"ap{i}", position=[float(p[0]), float(p[1]), gnd(p[0], p[1]) + AP_H]))
    lam = 3e8 / c["fc"]
    # trajectories: 50 % pedestrians (1.4 m/s), 50 % vehicles (10 m/s), SRS period 20 ms
    trajs = []; sg = StreetGraph(m, xs, ys)
    while len(trajs) < n_traj:
        veh = len(trajs) % 2 == 1; v = 10.0 if veh else 1.4
        t = traj_fast(sg, rng, v * 0.02, 400)
        if t is not None: trajs.append((t, veh))
    pos = np.concatenate([t for t, _ in trajs]); tid = np.concatenate([np.full(len(t), i) for i, (t, _) in enumerate(trajs)])
    veh = np.concatenate([np.full(len(t), v) for t, v in trajs]); tt = np.concatenate([np.arange(len(t)) * 0.02 for t, _ in trajs])
    print("positions", len(pos), "trajectories", len(trajs))
    freqs = mi.Float(np.linspace(-c["bw"] / 2, c["bw"] / 2, NF))
    solver = PathSolver(); B = 64; H = np.zeros((len(pos), NAP, 8, NF), np.complex64)
    t0 = time.time()
    for s in range(0, len(pos), B):
        for o in [o for o in scene.transmitters]: scene.remove(o)
        blk = pos[s:s + B]
        for j, p in enumerate(blk):
            scene.add(Transmitter(name=f"ue{j}", position=[float(p[0]), float(p[1]), gnd(p[0], p[1]) + UE_H]))
        paths = solver(scene, max_depth=3, los=True, specular_reflection=True, diffuse_reflection=False,
                       refraction=False, diffraction=True, synthetic_array=True, samples_per_src=100000,
                       max_num_paths_per_src=3000, seed=s)
        h = paths.cfr(frequencies=freqs, normalize_delays=False, out_type="numpy")  # (rx, rx_ant, tx, tx_ant, t, f)
        H[s:s + len(blk)] = np.transpose(h[:, :, :, 0, 0, :], (2, 0, 1, 3))
        if s // B % 10 == 0:
            print(f"  {s+len(blk)}/{len(pos)} {time.time()-t0:.0f}s", flush=True)
    pw = (np.abs(H) ** 2).mean((2, 3))
    print("zero-power share per AP", (pw == 0).mean(0).round(3), "all-zero share", (pw.sum(1) == 0).mean().round(4))
    np.savez(os.path.join(OUT, f"{name}.npz"), H=H, pos=pos.astype(np.float32), tid=tid, veh=veh, t=tt,
             ap=np.c_[ap, [gnd(a[0], a[1]) + AP_H for a in ap]], fc=c["fc"], bw=c["bw"], xs=xs, ys=ys, mask=m, gz=gz, ground=G)
    print("saved", name, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 240)
