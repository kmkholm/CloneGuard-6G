"""Post-hoc breakdown of the test predictions: AUC by speed class (ray-traced scenes) and by clone-distance bin,
for CloneGuard and the two reference tests, all seeds. -> results/breakdown.json"""
import os, json, glob, numpy as np
from sklearn.metrics import roc_auc_score
from data import load_rt, load_dichasus, split_by_traj, subset
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
EDGES = [0.25, 1, 5, 20, 100, 400]
out = {}
for name, split in (("dichasus", "test_day2"), ("dichasus", "test"), ("munich28", "test"), ("sf7", "test")):
    if name == "dichasus":
        veh = None
    else:
        D = load_rt(name); m = split_by_traj(D); veh = subset(D, m[2])["veh"]
    pattern = os.path.join(RES, "pred", f"{name}_main_s*_{split}.npz")
    for f in sorted(glob.glob(pattern)):
        p = np.load(f); y, d, a = p["y"], p["d"], p["a"]
        tag = os.path.basename(f).replace(".npz", "")
        r = {}
        for key, lab in (("s", "model"), ("corr", "corr"), ("pprof", "pprof")):
            s = p[key]
            r[lab] = dict(all=float(roc_auc_score(y, s)))
            if veh is not None:
                v = veh[a]
                r[lab]["ped"] = float(roc_auc_score(y[~v], s[~v])); r[lab]["veh"] = float(roc_auc_score(y[v], s[v]))
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                mm = (d >= lo) & (d < hi) & (y == 1); sel = (y == 0) | mm
                if mm.sum() > 50:
                    r[lab][f"d{lo}-{hi}"] = float(roc_auc_score(y[sel], s[sel]))
                    if veh is not None:
                        for cls, msk in (("ped", ~veh[a]), ("veh", veh[a])):
                            sel2 = sel & msk
                            if (y[sel2] == 1).sum() > 50:
                                r[lab][f"d{lo}-{hi}_{cls}"] = float(roc_auc_score(y[sel2], s[sel2]))
        out[tag] = r
json.dump(out, open(os.path.join(RES, "breakdown.json"), "w"), indent=1)
for k, v in out.items():
    print(k, {kk: round(vv["all"], 3) for kk, vv in v.items()}, "ped/veh", v["model"].get("ped"), v["model"].get("veh"))
