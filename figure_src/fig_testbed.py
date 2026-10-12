"""Figure 9: testbed timelines. (a) one same-gNB run: spoof logits of the victim's SUPI over time with the clone's
registration, detection, quarantine and ping outcomes; (b) detection latency per scenario and distance;
(c) legitimate-UE logits over a 15-min clone-free run (false-alarm check); (d) inference time / drops vs number of UEs."""
import os, sys, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); OUT = os.path.join(ROOT, "emulation", "out"); FIG = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7})
C = dict(legit="#0F4C5C", clone="#9E2A2B", q="#5F0F40", amb="#E9A23B")


def events(run):
    ev = []
    for f in glob.glob(os.path.join(OUT, run, "*", "events.jsonl")):
        for l in open(f):
            try: ev.append(json.loads(l))
            except Exception: pass
    return sorted(ev, key=lambda e: e["t"])


runs = sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(OUT, "*", "summary.json")) if not os.path.basename(os.path.dirname(f)).startswith("failed_"))
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0))
# (a) timeline of one run
r0 = [r for r in runs if r.startswith("same_d5")]
if r0:
    ev = events(r0[0]); s = json.load(open(os.path.join(OUT, r0[0], "summary.json")))
    cl = s["clones"][0]; supi = cl["supi"]; t0 = cl["t_pdu"] or cl["t_start"]
    sc = [e for e in ev if e["kind"] == "score" and e.get("supi") == supi]
    ax = axs[0, 0]
    for ran, col, lab, ms in ((None, C["legit"], "victim UE (RAN UE id 1, 1 of 20 scores logged)", 4), (91, C["clone"], "clone (RAN UE id 91), probation", 7)):
        pts = [(e["t"] - t0, e["logit"]) for e in sc if (e["ran_ue"] == 91) == (ran == 91)]
        if pts: ax.plot([p[0] for p in pts], [p[1] for p in pts], "o" if ran else ".", ms=ms, color=col, label=lab, zorder=5)
    thr = [e for e in ev if e["kind"] == "pla_up"][0]["thr"]; ax.axhline(thr, color="k", ls="--", lw=.8, label="threshold")
    if cl["t_detect"]: ax.axvline(cl["t_detect"] - t0, color=C["q"], lw=1.2, label=f"declared +{cl['t_detect']-t0:.2f} s")
    if cl["t_quarantine"]: ax.axvline(cl["t_quarantine"] - t0, color=C["amb"], lw=1.2, ls=":", label="quarantined")
    ax.axvline(0, color=C["clone"], lw=.8, ls="-.", label="clone PDU session")
    ax.set_xlabel("time from the clone's PDU session (s)"); ax.set_ylabel("spoof logit"); ax.set_title("(a) same-gNB clone at 5 m, seed 1, victim SUPI ...0001"); ax.legend(loc="center left", ncol=1, fontsize=6); ax.grid(alpha=.3)
    ax.set_xlim(-56, 16); ax.set_ylim(-9, 30)
# (b) detection latency vs scenario
ax = axs[0, 1]; lat = {}
for r in runs:
    s = json.load(open(os.path.join(OUT, r, "summary.json"))); c = json.load(open(os.path.join(OUT, r, "config.json")))
    if c["scenario"] == "none": continue
    key = f"same gNB\n{c['dist']} m" if c["scenario"] == "same_gnb" else "other gNB\n5 m"
    lat.setdefault(key, []).extend([x["latency_detect_s"] for x in s["clones"] if x.get("latency_detect_s") is not None])
if lat:
    keys = sorted(lat, key=lambda k: (k.startswith("other"), float(k.split()[-2]))); ax.boxplot([lat[k] for k in keys], tick_labels=keys, widths=.5, patch_artist=True, boxprops=dict(facecolor="#F3E4EC", color=C["q"]), medianprops=dict(color=C["q"]))
    for i, k in enumerate(keys): ax.plot(np.full(len(lat[k]), i + 1) + np.random.uniform(-.12, .12, len(lat[k])), lat[k], "o", ms=3, color=C["clone"], alpha=.7)
ax.set_ylabel("detection latency (s)"); ax.set_title("(b) time from clone PDU session to declaration"); ax.tick_params(axis="x", labelsize=6.5); ax.grid(alpha=.3, axis="y")
# (c) clone-free run
ax = axs[1, 0]; r1 = [r for r in runs if r.startswith("noclone")]
if r1:
    ev = events(r1[0]); t0 = min(e["t"] for e in ev if e["kind"] == "ue_pdu")
    sc = [e for e in ev if e["kind"] == "score"]
    for i, supi in enumerate(sorted({e["supi"] for e in sc})):
        pts = [(e["t"] - t0, e["logit"]) for e in sc if e["supi"] == supi]
        ax.plot([p[0] / 60 for p in pts], [p[1] for p in pts], ".", ms=2, alpha=.6, label=f"UE {i+1}")
    thr = [e for e in ev if e["kind"] == "pla_up"][0]["thr"]; ax.axhline(thr, color="k", ls="--", lw=.8, label="threshold")
    n_det = len([e for e in ev if e["kind"] == "detect"])
    ax.set_title(f"(c) clone-free run, 8 legitimate UEs: {n_det} false declarations"); ax.set_xlabel("time (min)"); ax.set_ylabel("spoof logit"); ax.legend(ncol=3, fontsize=6); ax.grid(alpha=.3)
# (d) load
ax = axs[1, 1]; pts = []
for r in runs:
    ev = events(r); c = json.load(open(os.path.join(OUT, r, "config.json")))
    sc = [e for e in ev if e["kind"] == "score"]
    if sc: pts.append((c["ues"] * c["sites"] + (0 if c["scenario"] == "none" else c["sites"]), np.mean([e["inf_ms"] for e in sc]), np.mean([e.get("batch", 1) for e in sc]), sc[-1].get("drop", 0) / max(sc[-1].get("rx", 1), 1)))
if pts:
    pts = np.array(pts); ax.plot(pts[:, 0], pts[:, 1], "o", color=C["q"], label="inference time per batch (ms)")
    ax2 = ax.twinx(); ax2.plot(pts[:, 0], 100 * pts[:, 3], "s", color=C["amb"], label="dropped snapshots (%)"); ax2.set_ylabel("dropped snapshots (%)")
    ax.set_xlabel("UE processes attached (legitimate + clones)"); ax.set_ylabel("inference time (ms)"); ax.set_title("(d) authenticator load (CPU, 4 threads)"); ax.legend(loc="center left"); ax2.legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "Figure9.png"), dpi=300); fig.savefig(os.path.join(FIG, "Figure9.pdf"))
print("Figure9 written from", len(runs), "runs")
