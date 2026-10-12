"""Results figures from results/*.json and results/pred/*.npz.
Figure 4: ROC curves (model vs reference tests) per test set + score distributions.
Figure 5: confusion matrices (main model, three test sets, seed 1).
Figure 6: detection rate vs clone distance (main vs references, three datasets) + ablation bars.
Figure 7: training curves (loss, validation AUC) for the three datasets.
Figure 8: t-SNE of snapshot embeddings (legit history vs clone queries) + AP pooling weights and
          memory attention on real DICHASUS episodes (XAI).
Also writes results/tables.json (means +- std over seeds) used by build_tables.py."""
import os, sys, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7})
DS = [("dichasus", "DICHASUS (measured, 1.27 GHz)"), ("munich28", "Munich 28 GHz (ray-traced)"), ("sf7", "San Francisco 7 GHz (ray-traced)")]
TEST = {"dichasus": "test_day2", "munich28": "test", "sf7": "test"}
C = dict(model="#5F0F40", corr="#0F4C5C", pprof="#E36414", legit="#0F4C5C", clone="#9E2A2B")


def load(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------------------------------------------------------- tables
tab = {}
for d, _ in DS:
    for tag in ["main"] + ["abl_rope", "abl_nopos", "abl_noforget", "abl_meanap", "abl_k4", "abl_k8", "abl_k32", "abl_nopred"]:
        seeds = [load(f"{d}_{tag}_s{s}") for s in (1, 2, 3)]; seeds = [x for x in seeds if x]
        if not seeds: continue
        for split in seeds[0]["test"]:
            for key in ("model", "corr_test", "power_profile_test"):
                for met in ("auc", "eer", "tpr_at_fpr1", "tpr_at_fpr01", "acc", "tpr", "fpr"):
                    v = [s["test"][split][key][met] for s in seeds]
                    tab[f"{d}|{tag}|{split}|{key}|{met}"] = dict(mean=float(np.mean(v)), std=float(np.std(v)), n=len(v))
            tab[f"{d}|{tag}|{split}|by_dist"] = [s["test"][split]["by_dist"] for s in seeds]
        tab[f"{d}|{tag}|params"] = seeds[0]["params"]; tab[f"{d}|{tag}|train_time_s"] = float(np.mean([s["train_time_s"] for s in seeds]))
        tab[f"{d}|{tag}|thr_fpr0.1"] = seeds[0].get("thr_fpr0.1")
json.dump(tab, open(os.path.join(RES, "tables.json"), "w"), indent=1)

# ---------------------------------------------------------------- Figure 4: ROC + score distributions
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6))
for j, (d, title) in enumerate(DS):
    p = os.path.join(RES, "pred", f"{d}_main_s1_{TEST[d]}.npz")
    ax = axs[0, j]; ax.set_title(title)
    if not os.path.exists(p):
        ax.text(.5, .5, "pending", ha="center"); continue
    e = np.load(p); y = e["y"]
    for key, lab, col in (("s", "CloneGuard", C["model"]), ("corr", "AP-weighted correlation test", C["corr"]), ("pprof", "AP power-profile test", C["pprof"])):
        fpr, tpr, _ = roc_curve(y, e[key]); ax.plot(fpr, tpr, color=col, lw=1.4, label=f"{lab} (AUC {roc_auc_score(y, e[key]):.3f})")
    ax.set_xscale("log"); ax.set_xlim(1e-4, 1); ax.set_ylim(0, 1.02); ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate"); ax.grid(alpha=.3); ax.legend(loc="lower right", fontsize=5.5, framealpha=.85)
    ax = axs[1, j]
    s = e["s"]; bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 60)
    ax.hist(s[y == 0], bins=bins, color=C["legit"], alpha=.65, density=True, label="legitimate continuation")
    ax.hist(s[y == 1], bins=bins, color=C["clone"], alpha=.65, density=True, label="clone (0.25 m - scene)")
    j0 = load(f"{d}_main_s1"); thr = j0["test"][TEST[d]]["model"]["thr"]
    ax.axvline(thr, color="k", ls="--", lw=1, label="validation EER threshold")
    if j0.get("thr_fpr0.1") is not None: ax.axvline(j0["thr_fpr0.1"], color="k", ls=":", lw=1, label="threshold at FPR 0.1 % (testbed)")
    ax.set_xlabel("spoof logit"); ax.set_ylabel("density"); ax.legend(loc="upper right", fontsize=5.5, framealpha=.85); ax.set_title(f"score distributions, {TEST[d].replace('_', ' ')}")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "Figure4.png"), dpi=300); fig.savefig(os.path.join(FIG, "Figure4.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Figure 5: confusion matrices
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2))
panels = [("dichasus", "test"), ("dichasus", "test_day2"), ("munich28", "test"), ("sf7", "test")]
for ax, (d, split) in zip(axs, panels):
    j0 = load(f"{d}_main_s1")
    if not j0 or split not in j0["test"]:
        ax.set_title(f"{d} {split} pending"); ax.axis("off"); continue
    cm = np.array(j0["test"][split]["model"]["cm"]); n = cm.sum(1, keepdims=True)
    ax.imshow(cm / n, cmap="Purples", vmin=0, vmax=1)
    for i in range(2):
        for k in range(2):
            ax.text(k, i, f"{cm[i,k]}\n({100*cm[i,k]/n[i,0]:.1f}%)", ha="center", va="center", fontsize=7, color="white" if cm[i, k] / n[i, 0] > .5 else "k")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["legit", "clone"]); ax.set_yticklabels(["legit", "clone"])
    ax.set_xlabel("decision"); ax.set_ylabel("truth"); ax.set_title({"dichasus": "DICHASUS", "munich28": "Munich 28 GHz", "sf7": "SF 7 GHz"}[d] + ", " + split.replace("_", " "))
fig.tight_layout(); fig.savefig(os.path.join(FIG, "Figure5.png"), dpi=300); fig.savefig(os.path.join(FIG, "Figure5.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Figure 6: distance curves + ablations
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
for j, (d, title) in enumerate(DS):
    ax = axs[j]; ax.set_title(title)
    p = os.path.join(RES, "pred", f"{d}_main_s1_{TEST[d]}.npz")
    if not os.path.exists(p): ax.text(.5, .5, "pending", ha="center"); continue
    e = np.load(p); y, dd = e["y"], e["d"]; j0 = load(f"{d}_main_s1")
    edges = [0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 400]
    for key, lab, col in (("s", "CloneGuard", C["model"]), ("corr", "correlation test", C["corr"]), ("pprof", "power-profile test", C["pprof"])):
        thr = j0["test"][TEST[d]]["model" if key == "s" else {"corr": "corr_test", "pprof": "power_profile_test"}[key]]["thr"]
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (y == 1) & (dd >= lo) & (dd < hi)
            if m.sum() > 30: xs.append(np.sqrt(lo * hi)); ys.append((e[key][m] >= thr).mean())
        ax.plot(xs, ys, "o-", color=col, lw=1.3, ms=3, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("clone-victim distance (m)"); ax.set_ylabel("detection rate at EER threshold"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
    if j == 0: ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "Figure6.png"), dpi=300); fig.savefig(os.path.join(FIG, "Figure6.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Figure 7: training curves
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4))
cols = {"dichasus": C["corr"], "munich28": C["pprof"], "sf7": C["model"]}
for d, title in DS:
    j0 = load(f"{d}_main_s1")
    if not j0: continue
    h = j0["history"]; st = [x["step"] for x in h]
    axs[0].plot(st, [x["loss"] for x in h], color=cols[d], lw=1, label=title)
    v = [(x["step"], x["val_auc"]) for x in h if "val_auc" in x]; axs[1].plot([a for a, _ in v], [b for _, b in v], "o-", color=cols[d], lw=1, ms=3, label=title)
axs[0].set_xlabel("training step"); axs[0].set_ylabel("training loss"); axs[0].set_yscale("log"); axs[0].legend(); axs[0].grid(alpha=.3)
axs[1].set_xlabel("training step"); axs[1].set_ylabel("validation AUC"); axs[1].set_ylim(0.9, 1.001); axs[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "Figure7.png"), dpi=300); fig.savefig(os.path.join(FIG, "Figure7.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Figure 8: t-SNE + XAI
from sklearn.manifold import TSNE
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
p = os.path.join(RES, "pred", "dichasus_main_s1_test_day2.npz")
if os.path.exists(p):
    e = np.load(p); z = e["z"][:3000]; y = e["y"][:3000]; dd = e["d"][:3000]
    emb = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(z)
    ax = axs[0]; sc = ax.scatter(emb[:, 0], emb[:, 1], s=3, c=e["s"][:3000], cmap="RdYlBu_r", vmin=-6, vmax=12, alpha=.8)
    ax.set_title("t-SNE, query embeddings", fontsize=8); ax.set_xticks([]); ax.set_yticks([]); plt.colorbar(sc, ax=ax, fraction=.04, label="episode spoof logit")
    # XAI: run the model on real episodes to get AP weights and memory attention
    sys.path.insert(0, os.path.join(ROOT, "code")); import torch
    from data import load_dichasus, EpisodeSampler, normalise
    from model import CloneGuard
    j0 = load("dichasus_main_s1"); a = j0["args"]; dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = load_dichasus("cf12"); S = EpisodeSampler(D, a["K"], dev, gaps=(1,)); A, M, Fb = D["H"].shape[1:]
    model = CloneGuard(A, M, Fb, d=a["d"], pos=a["pos"], forget=bool(a["forget"]), spatial=a["spatial"]).to(dev)
    model.load_state_dict(torch.load(os.path.join(ROOT, "runs", "dichasus_main_s1.pt"), map_location=dev)); model.eval()
    g = torch.Generator(device=dev); g.manual_seed(3); X, yy, dist, _ = S.sample(512, g, dist_hi=15.0, gap=1)
    with torch.no_grad():
        Xn, lp = normalise(X); logit, _, _ = model(Xn, lp)
        att = model.mem[-1].att.last_attn.mean(1)[:, -1].cpu().numpy(); pool = model.last_pool[:, -1].cpu().numpy()
    yy = yy.cpu().numpy()
    ax = axs[1]
    for lab, m, col in (("legitimate", yy == 0, C["legit"]), ("clone", yy == 1, C["clone"])):
        mu, sd = att[m].mean(0), att[m].std(0); xs = np.arange(-a["K"], 1)
        ax.plot(xs, mu, "o-", color=col, ms=3, lw=1.2, label=lab); ax.fill_between(xs, mu - sd, mu + sd, color=col, alpha=.2)
    ax.set_xlabel("memory position (0 = query)"); ax.set_ylabel("attention of the query"); ax.set_title("where the memory looks (last block)"); ax.legend(); ax.grid(alpha=.3)
    ax = axs[2]
    w_leg, w_cl = pool[yy == 0].mean(0), pool[yy == 1].mean(0); xs = np.arange(A)
    ax.bar(xs - .18, w_leg, .36, color=C["legit"], label="legitimate query"); ax.bar(xs + .18, w_cl, .36, color=C["clone"], label="clone query")
    ax.set_xticks(xs); ax.set_xticklabels([f"AP {k+1}" for k in xs]); ax.set_ylabel("AP pooling weight"); ax.set_title("which access points decide"); ax.set_ylim(0, 0.36); ax.legend(loc="upper right", fontsize=6)
    json.dump(dict(att_legit=att[yy == 0].mean(0).tolist(), att_clone=att[yy == 1].mean(0).tolist(), pool_legit=w_leg.tolist(), pool_clone=w_cl.tolist()),
              open(os.path.join(RES, "xai_dichasus.json"), "w"), indent=1)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "Figure8.png"), dpi=300); fig.savefig(os.path.join(FIG, "Figure8.pdf")); plt.close(fig)
print("figures 4-8 written")
