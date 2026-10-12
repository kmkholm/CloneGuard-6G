"""Figure 1 - CloneGuard overview, original design.
Layout (one 7.2 x 7.6 in page):
  A  "Where it happens": top-down map of the real DICHASUS hall (four arrays, measured victim trajectory coloured
     by time, clone position), with channel-snapshot cards hung on the trajectory by leader lines.
  B  "Per-snapshot encoder": the four AP cards of one real snapshot -> set attention drawn as a weighted complete
     graph (edge width = real attention weight) -> pooled embedding.
  C  "Channel memory": a time ribbon with K+1 memory slots; Householder path transforms as rotation glyphs between
     slots, the forget gate as a fading ribbon, and the real attention of the query drawn as drops on the slots,
     for a legitimate query and for a clone query.
  D  "Decision and response": a score gauge and the core-network actions with the measured testbed latencies.
  E  equation strip.
python fig1.py [run_name]   -> figures/Figure1.{pdf,png,tif}"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, Polygon, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "code"))
from data import load_dichasus, EpisodeSampler, normalise
from model import CloneGuard
OUT = os.path.join(ROOT, "figures"); os.makedirs(OUT, exist_ok=True)
RUN = sys.argv[1] if len(sys.argv) > 1 else "dichasus_main_s1"

# palette: deep navy ink, teal (legitimate / network), coral (clone / alarm), gold (memory), sand backgrounds
INK, MUTE, PAPER = "#15213B", "#6B7280", "#FFFFFF"
TEAL, TEAL_L, CORAL, CORAL_L, GOLD, GOLD_L, SLATE, SAND = "#127C8A", "#D7EEF1", "#D9534F", "#FBE3E2", "#C98A14", "#FBEFD5", "#3F4A66", "#F6F2EA"
TIME = LinearSegmentedColormap.from_list("time", ["#9BD3DA", "#127C8A", "#15213B"])
CSI = LinearSegmentedColormap.from_list("csi", ["#15213B", "#127C8A", "#F6F2EA", "#D9534F"])
ATT = LinearSegmentedColormap.from_list("att", ["#F6F2EA", "#F2C14E", "#D9534F", "#7A1F1C"])
plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})

W, H = 1000, 770
fig = plt.figure(figsize=(7.2, 7.2 * H / W)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def card(x, y, w, h, fc=PAPER, ec="#D9D4CA", lw=0.8, r=8, z=2, shadow=True):
    if shadow:
        ax.add_patch(FancyBboxPatch((x + 2.5, y + 3), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc="#000000", ec="none", alpha=0.08, zorder=z - 1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def txt(x, y, s, fs=7, c=INK, ha="center", va="center", z=8, **kw):
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, zorder=z, **kw)


def chip(x, y, w, h, fc, s, tc="white", fs=6.4, z=6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={h/2}", fc=fc, ec="none", zorder=z)); txt(x + w / 2, y + h / 2, s, fs, tc, z=z + 1)


def arrow(x0, y0, x1, y1, c=INK, lw=1.0, ms=8, ls="-", rad=0.0, z=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def heat(img, x, y, w, h, vmin=0, vmax=1, cmap=CSI, z=5, ec="#15213B"):
    ax.imshow(img, extent=[x, x + w, y + h, y], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, zorder=z, interpolation="nearest")
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=0.5, zorder=z + 1))


def head(x, y, letter, title, c):
    ax.add_patch(Circle((x + 9, y), 9, fc=c, ec="none", zorder=9)); txt(x + 9, y + 0.5, letter, 8, "white", z=10, fontweight="bold")
    txt(x + 24, y, title, 8.8, INK, ha="left", z=10, fontweight="bold")


# ------------------------------------------------------------ real data, real model, one real episode
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = load_dichasus("cf12"); spec = json.load(open(os.path.join(ROOT, "data", "dichasus", "spec_cf1x.json")))
cfg = json.load(open(os.path.join(ROOT, "results", RUN + ".json")))["args"]; K = cfg["K"]
A, M, Fb = D["H"].shape[1:]
model = CloneGuard(A, M, Fb, d=cfg["d"], pos=cfg["pos"], forget=bool(cfg["forget"]), spatial=cfg["spatial"]).to(dev)
model.load_state_dict(torch.load(os.path.join(ROOT, "runs", RUN + ".pt"), map_location=dev)); model.eval()
S = EpisodeSampler(D, K, dev, gaps=(1,)); g = torch.Generator(device=dev); g.manual_seed(11)
X, y, dist, anc = S.sample(256, g, dist_lo=3.0, dist_hi=8.0, gap=1)
with torch.no_grad():
    Xn, lp = normalise(X); logit, _, _ = model(Xn, lp)
    att = model.mem[-1].att.last_attn.mean(1)[:, -1].cpu().numpy()          # B, K+1 query attention over the memory
    pool = model.last_pool[:, -1].cpu().numpy()                               # B, A
    set_att = []
    # set-attention weights between AP tokens of the query snapshot (last set block, averaged heads)
    hq = model.tok(Xn[:, -1:], lp[:, -1:]).reshape(-1, A, cfg["d"])
    for blk in model.set[:-1]:
        hq = blk(hq)
    hn = model.set[-1].n1(hq); _, w_set = model.set[-1].att(hn, hn, hn, need_weights=True, average_attn_weights=True)
    w_set = w_set.cpu().numpy()
yy = y.cpu().numpy(); dd = dist.cpu().numpy(); pr = torch.sigmoid(logit).cpu().numpy(); anc = anc.cpu().numpy()
i_leg = int(np.nonzero(yy == 0)[0][0]); i_cl = int(np.nonzero(yy == 1)[0][np.argmin(np.abs(dd[yy == 1] - 5))])
Hn = np.abs(torch.complex(Xn[..., 0], Xn[..., 1]).cpu().numpy()); vmax = np.percentile(Hn[i_leg], 99.5)
a0 = anc[i_leg]; tid0 = D["tid"][a0]; seg = np.nonzero(D["tid"] == tid0)[0]; seg = seg[seg <= a0][-K - 1:]
traj = D["pos"][seg]; clone_pos = D["pos"][anc[i_cl]] if yy[i_cl] else None
# clone query position: the spoof index is not returned by the sampler; recover it by the stored distance
q_cl = None
cand = np.nonzero(np.abs(np.linalg.norm(D["pos"] - D["pos"][anc[i_cl]], axis=1) - dd[i_cl]) < 0.05)[0]
q_cl = D["pos"][cand[0]] if len(cand) else D["pos"][anc[i_cl]] + np.array([dd[i_cl], 0])

# ================================================================= A  scene (left, top)
AX, AY, AW, AH = 12, 30, 410, 344
card(AX, AY, AW, AH, fc=SAND, ec="#E4DED2"); head(AX + 10, AY + 16, "A", "a measured cell-free cluster", TEAL)
# map: hall extents from the recordings
P = D["pos"]; x0, x1, y0, y1 = P[:, 0].min() - 1.5, P[:, 0].max() + 1.5, P[:, 1].min() - 1.5, P[:, 1].max() + 1.5
MX, MY, MW, MH = AX + 24, AY + 40, 240, 230
sx = lambda v: MX + (v - x0) / (x1 - x0) * MW
sy = lambda v: MY + MH - (v - y0) / (y1 - y0) * MH
ax.add_patch(Rectangle((MX, MY), MW, MH, fc=PAPER, ec="#CFC8BC", lw=0.8, zorder=3))
ax.plot(sx(P[::15, 0]), sy(P[::15, 1]), ".", ms=0.8, color="#D5D0C6", zorder=4, rasterized=True)
for a in spec["antennas"]:
    px, py = sx(a["location"][0]), sy(a["location"][1]); d_ = np.array(a["direction"][:2]); d_ = d_ / np.linalg.norm(d_)
    ax.add_patch(Wedge((px, py), 26, np.degrees(np.arctan2(-d_[1], d_[0])) - 35, np.degrees(np.arctan2(-d_[1], d_[0])) + 35, fc=TEAL, alpha=0.12, ec="none", zorder=4))
    ax.add_patch(Rectangle((px - 7, py - 3), 14, 6, fc=TEAL, ec=INK, lw=0.5, zorder=6, angle=np.degrees(np.arctan2(-d_[1], d_[0])) + 90))
    txt(px, py - 12, "AP " + a["name"][-1], 6.2, TEAL, fontweight="bold")
pts = np.c_[sx(traj[:, 0]), sy(traj[:, 1])]; segs = np.stack([pts[:-1], pts[1:]], 1)
lc = LineCollection(segs, cmap=TIME, norm=plt.Normalize(0, len(segs)), lw=2.2, zorder=6); lc.set_array(np.arange(len(segs))); ax.add_collection(lc)
ax.plot(pts[-1, 0], pts[-1, 1], "o", ms=5, mfc=INK, mec="white", zorder=8)
ax.plot(sx(q_cl[0]), sy(q_cl[1]), "X", ms=8, mfc=CORAL, mec="white", mew=0.8, zorder=8)
ax.plot([pts[-1, 0], sx(q_cl[0])], [pts[-1, 1], sy(q_cl[1])], color=CORAL, lw=0.8, ls=(0, (3, 2)), zorder=7)
txt((pts[-1, 0] + sx(q_cl[0])) / 2, (pts[-1, 1] + sy(q_cl[1])) / 2 - 8, f"d = {dd[i_cl]:.1f} m", 6.2, CORAL, fontweight="bold")
txt(MX + MW / 2, MY + MH + 10, "Arena2036 hall, 1.27 GHz, 4 arrays x 8 antennas,", 5.4, MUTE); txt(MX + MW / 2, MY + MH + 20, "robot tracked by tachymeter (day 2, unseen in training)", 5.4, MUTE)
# snapshot cards hung on the trajectory (3 history snapshots + query + clone)
picks = [0, K // 2, K - 1]
for j, k in enumerate(picks):
    cx, cy = AX + 285, AY + 44 + j * 58
    heat(Hn[i_leg, k].reshape(A * M, Fb) ** 0.5, cx, cy, 108, 42, 0, vmax ** 0.5)
    ax.plot([pts[k, 0], cx], [pts[k, 1], cy + 21], color=TIME(k / K), lw=0.7, zorder=5)
    txt(cx + 54, cy + 48, f"snapshot t-{K - k}" if k < K - 1 else "snapshot t-1", 5.6, MUTE)
cx, cy = AX + 285, AY + 44 + 3 * 58
heat(Hn[i_leg, -1].reshape(A * M, Fb) ** 0.5, cx, cy, 108, 42, 0, vmax ** 0.5, ec=TEAL); ax.plot([pts[-1, 0], cx], [pts[-1, 1], cy + 21], color=INK, lw=0.7, zorder=5)
txt(cx + 54, cy + 48, "legitimate query t", 5.6, TEAL, fontweight="bold")
cx2, cy2 = AX + 285, AY + 44 + 4 * 58
heat(Hn[i_cl, -1].reshape(A * M, Fb) ** 0.5, cx2, cy2, 108, 42, 0, vmax ** 0.5, ec=CORAL); ax.plot([sx(q_cl[0]), cx2], [sy(q_cl[1]), cy2 + 21], color=CORAL, lw=0.7, ls=(0, (3, 2)), zorder=5)
txt(cx2 + 54, cy2 + 48, "clone query, same SUPI", 5.6, CORAL, fontweight="bold")
txt(AX + 339, AY + 36, r"$|\mathbf{H}|$ (rows AP x antenna, cols sub-band)", 5.0, MUTE)

# ================================================================= B  per-snapshot encoder (right, top)
BX, BY, BW, BH = 436, 30, 552, 160
card(BX, BY, BW, BH, fc=PAPER, ec="#E4DED2"); head(BX + 10, BY + 16, "B", "per-snapshot encoder: AP tokens, set attention, pooling", TEAL)
APC = [TEAL, "#3A9BA8", GOLD, CORAL]
for a_ in range(A):
    cx, cy = BX + 18 + a_ * 58, BY + 44
    heat(Hn[i_leg, -1, a_] ** 0.5, cx, cy, 48, 40, 0, vmax ** 0.5, ec=APC[a_]); txt(cx + 24, cy + 48, f"AP {a_+1}", 5.8, APC[a_], fontweight="bold")
    chip(cx + 2, cy + 56, 44, 11, SLATE, "CNN", fs=5.4)
    txt(cx + 24, cy + 76, f"$p_{{t,{a_+1}}}$ = {lp[i_leg, -1, a_].item():.0f} dB", 5.2, MUTE)
txt(BX + 128, BY + 134, "unit energy, per-AP phase reference, AP embedding", 5.2, MUTE)
txt(BX + 128, BY + 146, r"$\mathbf{u}_{t,a}=\mathrm{CNN}(\tilde{\mathbf{H}}_{t,a})+\mathbf{e}_a+\phi(p_{t,a})$", 6.2, INK)
# set-attention graph with real weights
GX, GY, R = BX + 318, BY + 92, 38
pos4 = [(GX - R, GY - R * 0.55), (GX + R, GY - R * 0.55), (GX + R, GY + R * 0.55), (GX - R, GY + R * 0.55)]
w4 = w_set[i_leg]
for i in range(A):
    for j in range(A):
        if i == j: continue
        lw = 0.3 + 4.0 * w4[i, j] / w4.max()
        ax.plot([pos4[i][0], pos4[j][0]], [pos4[i][1], pos4[j][1]], color=APC[i], lw=lw, alpha=0.55, zorder=5, solid_capstyle="round")
for i in range(A):
    ax.add_patch(Circle(pos4[i], 10, fc=APC[i], ec="white", lw=1.2, zorder=7)); txt(pos4[i][0], pos4[i][1] + 0.5, f"{i+1}", 6.5, "white", z=8, fontweight="bold")
txt(GX, GY - R - 22, "set attention (2 blocks, 4 heads)", 6.2, INK, fontweight="bold"); txt(GX, GY - R - 12, "edge width = real attention weight", 5.2, MUTE)
txt(GX, GY + R + 18, "any number or placement of APs", 5.2, MUTE)
arrow(BX + 250, BY + 92, GX - R - 14, BY + 92, INK, 1.0)
# pooling
PX, PY = BX + 430, BY + 60
arrow(GX + R + 14, GY, PX - 6, GY, INK, 1.0)
txt(PX + 50, PY - 8, "attention pooling", 6.4, INK, fontweight="bold")
for a_ in range(A):
    wv = pool[i_leg, a_] / pool[i_leg].max()
    ax.add_patch(Rectangle((PX, PY + 4 + a_ * 13), 70 * wv, 9, fc=APC[a_], ec="none", zorder=5)); txt(PX + 74, PY + 8.5 + a_ * 13, f"{pool[i_leg, a_]:.2f}", 5.4, MUTE, ha="left")
txt(PX + 50, PY + 66, r"$\mathbf{z}_t=\sum_a w_{t,a}\,\mathbf{h}_{t,a}\in\mathbb{R}^{128}$", 6.4, INK)
ax.add_patch(FancyBboxPatch((PX + 22, PY + 78), 56, 16, boxstyle="round,pad=0,rounding_size=5", fc=TEAL, ec="none", zorder=6)); txt(PX + 50, PY + 86, "embedding", 6.0, "white", z=7)

# ================================================================= C  channel memory (right, middle)
CX, CY, CW, CH = 436, 204, 552, 300
card(CX, CY, CW, CH, fc=PAPER, ec="#E4DED2"); head(CX + 10, CY + 16, "C", "channel memory: path-encoded forgetting attention", GOLD)
T = K + 1
RX0, RX1 = CX + 92, CX + 452
xs_slot = np.linspace(RX0, RX1, T)
for row, (lab, idx, col, yrow) in enumerate((("legitimate query", i_leg, TEAL, CY + 86), ("clone query", i_cl, CORAL, CY + 196))):
    # fading ribbon = cumulative forget gate (illustrated by the attention mass), slots, path glyphs
    ribbon = np.linspace(0.18, 1.0, T)
    for t in range(T - 1):
        ax.add_patch(Rectangle((xs_slot[t], yrow - 9), xs_slot[t + 1] - xs_slot[t], 18, fc=GOLD, alpha=0.08 + 0.22 * ribbon[t], ec="none", zorder=3))
    aa = att[idx]
    for t in range(T):
        rr = 5 + 9 * (aa[t] / aa.max())
        ax.add_patch(Circle((xs_slot[t], yrow), rr, fc=ATT(aa[t] / aa.max()), ec=INK if t == T - 1 else "none", lw=1.2 if t == T - 1 else 0, zorder=6))
        if t < T - 1 and t % 4 == 0:
            # Householder glyph between slots: a small rotation arc with a tick
            mx = (xs_slot[t] + xs_slot[t + 1]) / 2
            ax.add_patch(Wedge((mx, yrow - 20), 7, 200, 340, width=2.2, fc=GOLD, ec="none", zorder=6)); txt(mx, yrow - 31, r"$\mathbf{H}_s$", 5.4, GOLD)
    ax.plot(xs_slot[-1], yrow, marker="*", ms=0, zorder=7)
    txt(RX0 - 8, yrow, lab, 6.2, col, ha="right", fontweight="bold")
    txt(RX1 + 22, yrow - 5, r"$\ell_t$ = %.1f" % logit[idx].item(), 6.4, col, ha="left", fontweight="bold"); txt(RX1 + 22, yrow + 7, f"p = {pr[idx]:.2f}", 5.6, col, ha="left")
    txt(xs_slot[0], yrow + 22, "t-K", 5.6, MUTE); txt(xs_slot[-2], yrow + 22, "t-1", 5.6, MUTE); txt(xs_slot[-1], yrow + 22, "query", 5.8, INK, fontweight="bold")
    # distribution of the query attention as a small bar strip under the slots (how much mass per slot)
    for t in range(T):
        ax.add_patch(Rectangle((xs_slot[t] - 5, yrow + 30), 10, 22 * aa[t] / aa.max(), fc=col, alpha=0.75, ec="none", zorder=5))
    txt(RX0 - 8, yrow + 40, "query attention", 5.2, MUTE, ha="right")
txt(CX + 276, CY + 42, "ribbon = forget-gate retention; drop size and colour = query attention on that memory slot (real episode, day 2)", 5.2, MUTE)
txt(CX + 276, CY + 150, "a clone query is not explained by the path of recent channel changes: the attention pattern shifts and the spoof head fires", 5.0, CORAL)
txt(CX + 276, CY + 264, "heads on the last two tokens: spoof logit $\\ell_t=g([\\tilde{\\mathbf{x}}_t;\\,\\tilde{\\mathbf{x}}_t-\\tilde{\\mathbf{x}}_{t-1}])$, prediction $\\hat{\\mathbf{z}}_t=W_p\\tilde{\\mathbf{x}}_{t-1}$", 5.8, INK)
txt(CX + 276, CY + 282, "no attacker data: query = next snapshot of the user (y=0) or a snapshot at distance d (y=1); gaps 1-16", 5.2, MUTE)
arrow(PX + 50, PY + 94, PX + 50, CY - 2, INK, 1.0)

# ================================================================= D  decision and response (left, middle)
DX, DY, DW, DH = 12, 386, 410, 120
card(DX, DY, DW, DH, fc=PAPER, ec="#E4DED2"); head(DX + 10, DY + 16, "D", "decision and core-network response", CORAL)
# gauge
GX2, GY2 = DX + 62, DY + 92
for k_, (a0_, a1_, cc) in enumerate(((180, 118, TEAL), (118, 62, GOLD), (62, 0, CORAL))):
    ax.add_patch(Wedge((GX2, GY2), 46, a1_, a0_, width=12, fc=cc, ec="white", lw=1, zorder=5))
thr_ang = 118
for lab_, val_, cc in (("legit", logit[i_leg].item(), TEAL), ("clone", logit[i_cl].item(), CORAL)):
    ang = np.radians(180 - 180 * np.clip((val_ + 8) / 28, 0, 1))
    ax.plot([GX2, GX2 + 40 * np.cos(ang)], [GY2, GY2 - 40 * np.sin(ang)], color=cc, lw=2.2, zorder=7, solid_capstyle="round")
txt(GX2 - 40, GY2 - 2, "accept", 5.0, TEAL); txt(GX2 + 40, GY2 - 2, "alarm", 5.0, CORAL); txt(GX2 + 2, GY2 - 54, r"$\tau$ = score at 0.1 % FPR", 5.2, MUTE)
txt(GX2, GY2 + 16, "spoof logit per SRS", 5.4, INK)
# rule + actions
chip(DX + 128, DY + 34, 130, 14, TEAL, r"$\ell_t<\tau$: accept, slide memory", fs=5.4)
chip(DX + 128, DY + 53, 130, 14, GOLD, r"$\ell_t\geq\tau$: alarm, freeze memory", fs=5.4)
chip(DX + 128, DY + 72, 130, 14, CORAL, "3 alarms in a row: declare", fs=5.4)
arrow(DX + 258, DY + 79, DX + 268, DY + 79, INK, 0.9)
for j, (tx, cc, lat) in enumerate((("UPF drops PDU session", CORAL, "9 ms"), ("UDR withdraws subscription", SLATE, "90 ms"), ("AMF rejects re-registration", INK, "cause 11"))):
    chip(DX + 266, DY + 34 + j * 19, 108, 13, cc, tx, fs=4.5); txt(DX + 405, DY + 40.5 + j * 19, lat, 4.8, MUTE, ha="right")
txt(DX + 266, DY + 104, "Open5GS / UERANSIM testbed: declaration 0.4 s after the clone's PDU session", 5.0, MUTE)


# ================================================================= E  equation strip (bottom)
EX, EY, EW, EH = 12, 518, 976, 108
card(EX, EY, EW, EH, fc=SAND, ec="#E4DED2"); head(EX + 10, EY + 16, "E", "the memory step, in one line", SLATE)
txt(EX + 488, EY + 44, r"$a_{t,s}=\dfrac{\mathbf{q}_t^{\top}\,\mathbf{H}_t\mathbf{H}_{t-1}\cdots\mathbf{H}_{s+1}\,\mathbf{k}_s}{\sqrt{d_h}}\;+\;\sum_{r=s+1}^{t}\log f_r\,,\qquad \mathbf{H}_s=\mathbf{I}-\beta_s\mathbf{w}_s\mathbf{w}_s^{\top},\;\;\beta_s=2\sigma(\mathbf{a}^{\top}\mathbf{x}_s),\;\;\|\mathbf{w}_s\|=1,\qquad f_r=\sigma(\mathbf{w}_f^{\top}\mathbf{x}_r)$", 7.4, INK)
txt(EX + 488, EY + 74, r"$\alpha_{t,s}=\mathrm{softmax}_{s\leq t}(a_{t,s}),\quad \mathbf{x}_t\leftarrow\mathbf{x}_t+W_o\sum_{s\leq t}\alpha_{t,s}\mathbf{v}_s,\qquad \mathcal{L}=\mathrm{BCE}(\ell_t,y)+\lambda\,(1-\cos(\hat{\mathbf{z}}_t,\mathbf{z}_t))\,\mathbb{1}[y=0]$", 7.0, INK)
txt(EX + 488, EY + 90, "the key of an old snapshot is carried to the query through the Householder transforms of every snapshot between them,", 5.2, MUTE); txt(EX + 488, EY + 99, "so the logit follows the path of channel changes rather than elapsed time; the forget term discounts snapshots behind abrupt changes", 5.2, MUTE)

# ================================================================= F  compared variants + datasets (bottom strip)
FX, FY, FW, FH = 12, 638, 976, 108
card(FX, FY, FW, FH, fc=PAPER, ec="#E4DED2"); head(FX + 10, FY + 16, "F", "evaluation: three datasets, two reference tests, eight ablations, one protocol-level testbed", SLATE)
cols3 = [("DICHASUS cell-free (measured)", "1.27 GHz, 4 x 8 ant., 2 days, 68k snapshots", TEAL), ("Munich 28 GHz (Sionna RT)", "6 APs x 8 ant., 240 trajectories, FR2", GOLD), ("San Francisco 7 GHz (Sionna RT)", "6 APs x 8 ant., 240 trajectories, FR3", CORAL)]
for j, (nm, ds, cc) in enumerate(cols3):
    chip(FX + 16 + j * 200, FY + 34, 186, 15, cc, nm, fs=5.8); txt(FX + 109 + j * 200, FY + 60, ds, 5.0, MUTE)
chip(FX + 616, FY + 34, 170, 15, SLATE, "reference tests (training-free)", fs=5.8); txt(FX + 701, FY + 60, "AP-weighted correlation, AP power profile", 5.0, MUTE)
chip(FX + 800, FY + 34, 170, 15, INK, "ablations (same episodes, seed)", fs=5.8); txt(FX + 885, FY + 60, "RoPE / no positions, no forget gate,", 5.0, MUTE); txt(FX + 885, FY + 70, "mean AP pooling, K = 4/8/32, no prediction loss", 5.0, MUTE)
txt(FX + 488, FY + 90, "clone distance 0.25 m - 400 m, trajectory-disjoint splits; day 2 of the measurements is used only for testing and for the testbed replay", 5.2, MUTE)

fig.savefig(os.path.join(OUT, "Figure1.pdf")); fig.savefig(os.path.join(OUT, "Figure1.png"), dpi=600)
fig.savefig(os.path.join(OUT, "Figure1.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
print("Figure 1 written", RUN, "legit", logit[i_leg].item(), "clone", logit[i_cl].item(), "d", dd[i_cl])
