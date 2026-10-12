"""Figure 2: the three channel datasets. (a) DICHASUS Arena2036 cell-free layout with the four arrays and the
robot trajectories (day 1 train/test, day 2 held out); (b) Munich 28 GHz and (c) San Francisco 7 GHz ray-traced
scenes: street mask, lamp-post APs, sample trajectories; (d) per-snapshot temporal correlation vs lag per dataset;
(e) SNR distribution of the ray-traced links.  Output figures/Figure2.png/.pdf"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "code"))
from data import load_dichasus, load_rt
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
C = dict(train="#1b6ca8", test="#e07b39", day2="#2a9d8f", ap="#c0392b", ped="#6c5ce7", veh="#f39c12")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7})


def corr_vs_lag(D, lags, n=300, seed=0):
    H = D["H"].reshape(len(D["H"]), -1); tid = D["tid"]
    rng = np.random.default_rng(seed); out = []
    for lag in lags:
        ok = np.nonzero((np.arange(len(tid) - lag) >= 0) & (tid[lag:] == tid[:-lag]))[0]
        idx = rng.choice(ok, min(n, len(ok)), replace=False)
        a, b = H[idx], H[idx + lag]
        c = np.abs((a * b.conj()).sum(1)) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        out.append(c.mean())
    return np.array(out)


fig = plt.figure(figsize=(7.2, 6.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
# (a) DICHASUS
ax = fig.add_subplot(gs[0, 0])
spec = json.load(open(os.path.join(ROOT, "data", "dichasus", "spec_cf0x.json")))
for name, col, lab in (("cf02", C["train"], "cf02 (day 1, train/val)"), ("cf03", C["test"], "cf03 (day 1, test)"), ("cf12", C["day2"], "cf12 (day 2, test, testbed)")):
    D = load_dichasus(name); p = D["pos"][::10]
    ax.plot(p[:, 0], p[:, 1], ".", ms=1.2, color=col, alpha=.6, label=lab, rasterized=True)
for a in spec["antennas"]:
    ax.plot(a["location"][0], a["location"][1], "^", ms=9, color=C["ap"], mec="k")
    ax.annotate(a["name"][-1], (a["location"][0], a["location"][1]), xytext=(4, 4), textcoords="offset points", fontsize=8, weight="bold")
ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title("(a) DICHASUS, 1.27 GHz (measured)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1, markerscale=6, frameon=False, fontsize=6)
# (b)(c) ray-traced scenes
for j, (name, title) in enumerate((("munich28", "(b) Munich, 28 GHz (RT)"), ("sf7", "(c) San Francisco, 7 GHz (RT)"))):
    ax = fig.add_subplot(gs[0, 1 + j])
    f = os.path.join(ROOT, "data", "rt", f"{name}.npz")
    if not os.path.exists(f):
        ax.set_title(title + " [pending]"); continue
    d = np.load(f)
    xs, ys, m = d["xs"], d["ys"], d["mask"]
    ax.imshow(np.where(m.T, 1.0, 0.25), origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="gray", vmin=0, vmax=1.15)
    tid, pos, veh = d["tid"], d["pos"], d["veh"]
    for t in np.unique(tid)[:24]:
        p = pos[tid == t]; ax.plot(p[:, 0], p[:, 1], lw=1.0, color=C["veh"] if veh[tid == t][0] else C["ped"], alpha=.9)
    ap = d["ap"]; ax.plot(ap[:, 0], ap[:, 1], "^", ms=8, color=C["ap"], mec="k")
    for k, a in enumerate(ap): ax.annotate(f"AP{k+1}", (a[0], a[1]), xytext=(4, 3), textcoords="offset points", fontsize=7, weight="bold")
    ax.plot([], [], color=C["ped"], label="pedestrian 1.4 m/s"); ax.plot([], [], color=C["veh"], label="vehicle 10 m/s")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1, frameon=False, fontsize=6); ax.set_title(title); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
# (d) correlation vs lag
ax = fig.add_subplot(gs[1, 0:2])
lags = [1, 2, 4, 8, 16, 32]
rows = []
for name, lab, col in (("dichasus", "DICHASUS (20 Hz, robot 0.26 m/s)", C["train"]), ("munich28", "Munich 28 GHz (SRS 50 Hz)", C["test"]), ("sf7", "San Francisco 7 GHz (SRS 50 Hz)", C["day2"])):
    if name == "dichasus":
        D = load_dichasus("cf02"); c = corr_vs_lag(D, lags); ax.plot(lags, c, "o-", color=col, label=lab); rows.append((lab, c))
    elif os.path.exists(os.path.join(ROOT, "data", "rt", f"{name}.npz")):
        D = load_rt(name)
        for v, ls, tag in ((False, "-", "pedestrian"), (True, "--", "vehicle")):
            m = D["veh"] == v; Dv = dict(H=D["H"][m], tid=D["tid"][m]); c = corr_vs_lag(Dv, lags)
            ax.plot(lags, c, "o" + ls, color=col, label=f"{lab}, {tag}"); rows.append((lab + " " + tag, c))
ax.set_xscale("log", base=2); ax.set_xticks(lags); ax.set_xticklabels(lags); ax.set_xlabel("lag (snapshots)")
ax.set_ylabel("|normalised correlation|"); ax.set_title("(d) temporal channel correlation of a legitimate user"); ax.legend(frameon=False, ncol=1, loc="lower left", fontsize=5.5); ax.grid(alpha=.3)
# (e) SNR
ax = fig.add_subplot(gs[1, 2])
for name, col in (("munich28", C["test"]), ("sf7", C["day2"])):
    f = os.path.join(ROOT, "data", "rt", f"{name}.npz")
    if not os.path.exists(f): continue
    D = load_rt(name); p = (np.abs(D["H"]) ** 2).mean((2, 3)).sum(1); snr = 10 * np.log10(p / D["noise"] + 1e-30)
    ax.hist(snr[snr > -20], bins=60, color=col, alpha=.6, label=name.replace("munich28", "Munich 28 GHz").replace("sf7", "SF 7 GHz"), density=True)
ax.set_xlabel("sum SNR over APs (dB)"); ax.set_ylabel("density"); ax.set_title("(e) link budget, ray-traced"); ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "Figure2.png"), dpi=300); fig.savefig(os.path.join(ROOT, "figures", "Figure2.pdf"))
json.dump({k: v.tolist() for k, v in rows}, open(os.path.join(ROOT, "figures", "fig2_corr.json"), "w"), indent=1)
print("saved Figure2")
