"""Figure 3: the protocol-level testbed. (a) logical topology (nodes, segments, the CSI side channel and the control
path); (b) GNS3 screenshot (figures/gns3_shot.png); (c) clone-attack sequence."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); FIG = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.family": "Arial", "font.size": 7})
INK, MUTE = "#15213B", "#6B7280"
C = dict(core="#3D6B4F", gnb="#127C8A", ue="#5B6478", clone="#D9534F", pla="#5F0F40", seg="#C9C2B6", csi="#E36414")
fig = plt.figure(figsize=(7.2, 7.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.05, 1])
ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")


def box(x, y, w, h, fc, text, tc="white", fs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=fc, ec="none", zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, zorder=4, linespacing=1.15)


ax.text(0, 49, "(a) Testbed topology (GNS3 VM, Docker nodes)", fontsize=8.5, weight="bold", va="top")
box(2, 36, 34, 9, C["core"], "5G core node (Open5GS 2.7.2 + MongoDB)\nNRF, SCP, AUSF, UDM, UDR, PCF, BSF, NSSF, AMF, SMF, UPF\ncontrol agent: PDU cut (iptables), subscription withdrawal", fs=5.4)
box(64, 36, 34, 9, C["pla"], "Authenticator node (CloneGuard, PyTorch CPU)\nper-SUPI channel memory, batched scoring,\nAMF/SMF event feed, quarantine commands", fs=5.9)
ax.annotate("", xy=(36, 42.5), xytext=(64, 42.5), arrowprops=dict(arrowstyle="-|>", color=C["pla"], lw=1.0), zorder=5); ax.text(50, 43.4, "quarantine(SUPI)", color=C["pla"], fontsize=6, ha="center")
ax.annotate("", xy=(64, 38.5), xytext=(36, 38.5), arrowprops=dict(arrowstyle="-|>", color=C["core"], lw=1.0), zorder=5); ax.text(50, 36.6, "AMF / SMF events: SUPI, RAN UE id, PDU IP", color=C["core"], fontsize=6, ha="center")
ax.add_patch(FancyBboxPatch((2, 30), 96, 1.2, boxstyle="round,pad=0.2", fc=C["seg"], ec="none", zorder=2))
ax.text(50, 32.2, "core segment 10.60.0.0/24: N2 (NGAP/SCTP), N3 (GTP-U), CSI side channel, control; 2 ms backhaul delay", ha="center", fontsize=6, color=MUTE)
ax.plot([19, 19], [36, 31.2], color=INK, lw=0.8, zorder=2); ax.plot([81, 81], [36, 31.2], color=INK, lw=0.8, zorder=2)
for k in range(4):
    x = 3 + k * 24.5
    ax.plot([x + 10, x + 10], [30, 24], color=INK, lw=0.8, zorder=2)
    box(x, 18, 20, 6, C["gnb"], f"gNB {k+1} (UERANSIM nr-gnb)\nNG setup, RRC, NAS relay\nCSI replay of measured segments", fs=5.5)
    ax.add_patch(FancyBboxPatch((x + 1, 13), 18, 1.0, boxstyle="round,pad=0.2", fc=C["seg"], ec="none", zorder=2))
    ax.text(x + 10, 11.3, f"radio segment 10.70.{k+1}.0/24", ha="center", fontsize=5.4, color=MUTE)
    ax.plot([x + 10, x + 10], [18, 14], color=INK, lw=0.8, zorder=2)
    box(x, 3, 9.2, 6.2, C["ue"], f"UE node {k+1}\n2 UEs (nr-ue)\nprovisioned K, OPc", fs=5.2)
    box(x + 10.8, 3, 9.2, 6.2, C["clone"], f"clone {k+1}\nvictim's SUPI,\nK and OPc", fs=5.2)
    ax.plot([x + 4.6, x + 4.6], [13, 9.2], color=INK, lw=0.8, zorder=2); ax.plot([x + 15.4, x + 15.4], [13, 9.2], color=INK, lw=0.8, zorder=2)
ax.annotate("", xy=(78, 36), xytext=(21, 24), arrowprops=dict(arrowstyle="-|>", color=C["csi"], lw=1.0, ls="--", connectionstyle="arc3,rad=-0.22"), zorder=5)
ax.text(49, 27.3, "uplink CSI snapshots of every attached UE (UDP, 10/s per UE, 8 KB each)", color=C["csi"], fontsize=6, ha="center")
ax.text(50, 0.6, "every UE process is bound at its gNB to a trajectory segment of DICHASUS day 2 (never used for training); a clone is bound to a segment that starts $d$ m from the victim", fontsize=5.6, color=MUTE, ha="center")

# (b) screenshot
ax = fig.add_subplot(gs[1, 0]); ax.axis("off"); ax.set_title("(b) GNS3 project while a run is active (4 sites, 19 nodes)", fontsize=8, loc="left")
shot = os.path.join(FIG, "gns3_shot.png")
if os.path.exists(shot):
    ax.imshow(plt.imread(shot))
else:
    ax.text(.5, .5, "screenshot pending", ha="center", transform=ax.transAxes)

# (c) sequence
ax = fig.add_subplot(gs[1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off"); ax.set_title("(c) Clone attack and response (measured latencies)", fontsize=8, loc="left")
lanes = [("victim UE", 8, C["ue"]), ("clone", 26, C["clone"]), ("gNB", 46, C["gnb"]), ("AMF/UDR/UPF", 68, C["core"]), ("authenticator", 92, C["pla"])]
for nm, x, cc in lanes:
    ax.text(x, 97, nm, ha="center", fontsize=6.0, color=cc, weight="bold"); ax.plot([x, x], [2, 93], color=cc, lw=0.7, alpha=.5)
msgs = [(8, 68, 88, "registration (5G-AKA ok), PDU session", INK), (46, 92, 81, "CSI of RAN UE 1 every 100 ms", C["csi"]),
        (92, 92, 74, "memory of the SUPI slides (accepted)", C["pla"]), (26, 68, 66, "registration with the same SUPI: AKA ok, PDU", C["clone"]),
        (68, 8, 59, "Service Reject (identity cannot be derived)", C["core"]), (46, 92, 52, "CSI of RAN UE 91 (another place)", C["csi"]),
        (92, 92, 45, "3 alarms in a row: clone declared (0.4 s)", C["pla"]), (92, 68, 38, "quarantine(SUPI)", C["pla"]),
        (68, 68, 31, "UPF drops the PDU (9 ms); UDR withdraws the subscription (90 ms)", C["core"]),
        (26, 68, 23, "re-registration attempt", C["clone"]), (68, 26, 16, "Registration Reject (cause 11)", C["core"]),
        (8, 68, 9, "victim re-registers: rejected until the credential is restored", C["ue"])]
for x0, x1, y, tx, cc in msgs:
    if x0 == x1:
        ax.annotate("", xy=(x0 - 7, y - 2.5), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=cc, lw=0.8, connectionstyle="arc3,rad=0.6"))
        ax.text(x0 - 3, y + 2.4, tx, ha="right", fontsize=5.4, color=cc)
    else:
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=cc, lw=0.8))
        ax.text((x0 + x1) / 2, y + 2.4, tx, ha="center", fontsize=5.4, color=cc)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "Figure3.png"), dpi=300); fig.savefig(os.path.join(FIG, "Figure3.pdf"))
print("Figure3 written")
