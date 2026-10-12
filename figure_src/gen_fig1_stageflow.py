"""Figure 1 (draw.io, paper2drawio-v2 style): CloneGuard methodology.
Inventory (from code/model.py, code/data.py, code/train.py, emulation/node.py):
  Input: uplink snapshot H_t in C^{A x M x F} per SRS (DICHASUS A=4, M=8, F=64; ray-traced A=6, M=8, F=32)
  Normalisation: unit energy, per-AP phase reference (strongest element), AP power share p_{t,a} (dB)
  Stage 2 tokenizer: 3 conv layers (3x5; 32, 64, 64 ch; strides (1,2),(2,2)), AdaptiveAvgPool 2x4, Linear 512->128,
           + AP embedding e_a + Linear(p_{t,a}); set attention x2 blocks (4 heads, pre-norm, FFN 2d); attention pooling
  Stage 3: memory Z = [z_{t-K} ... z_{t-1}, z_t + q_flag], K = 16, d = 128
  Stage 4: 2 memory blocks, 4 heads: PaTH keys (Householder H_s = I - beta_s w_s w_s^T, beta_s = 2 sigma(a^T x_s)),
           forget gate f_s = sigma(w_f^T x_s), logits a_ts = q_t^T H_t...H_{s+1} k_s / sqrt(d_h) + sum log f_r, causal
           softmax, output projection, residual, pre-norm FFN (128 -> 256 -> 128)
  Stage 5: head g on [x_t ; x_t - x_{t-1}] (256 -> 128 GELU -> 1) = spoof logit; prediction head W_p (128 -> 128)
           loss BCE + 0.1 (1 - cos(z_hat, z)) on legitimate episodes; 775 k parameters
  Stage 6: accept if l < tau (memory slides), alarm if l >= tau (memory frozen), m = 3 alarms -> declare;
           core: UPF PDU drop, UDR subscription withdrawal, AMF rejects re-registration
python gen_fig1.py -> figures/drawio/Figure1.drawio (+ png/pdf via draw.io desktop)"""
import os, subprocess
from figlib import Fig, R_, it, sub
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "figures", "drawio"); os.makedirs(OUT, exist_ok=True)
f = Fig(page_w=1560, page_h=700, scale=0.6, name="Figure1")

# ------------------------------------------------------------------ Stage 1: input
f.stage(8, 10, 760, 300, "Stage 1: Uplink Channel Snapshots", "green")
f.box(20, 75, 250, 95, "Cell-free cluster<br>" + it("A") + " access points × " + it("M") + " antennas", "gray")
f.box(20, 190, 250, 95, "Sounding reference signal<br>every " + it("T") + "<sub>SRS</sub>", "gray")
f.arrow([(270, 122), (300, 122), (300, 180)]); f.arrow([(270, 237), (300, 237), (300, 180)], head=False)
f.arrow([(300, 180), (330, 180)])
f.box(330, 135, 180, 90, it("H") + "<sub>" + it("t") + "</sub> ∈ ℂ<sup>" + it("A") + "×" + it("M") + "×" + it("F") + "</sup>", "white", fs=18)
f.arrow([(510, 180), (545, 180)])
f.box(545, 90, 210, 180, "<div style='text-align:left'>• unit energy<br>• per-AP phase reference<br>• AP power share<br>&nbsp;&nbsp;" + it("p") + "<sub>" + it("t,a") + "</sub> (dB)</div>", "lightgreen", fs=16)
f.arrow([(300, 310), (300, 352)])
f.output_label(310, 312, 380, 40, it("H̃") + "<sub>" + it("t") + "</sub> ∈ ℂ<sup>" + it("A") + "×" + it("M") + "×" + it("F") + "</sup>, " + it("p") + "<sub>" + it("t") + "</sub> ∈ " + R_(it("A")))

# ------------------------------------------------------------------ Stage 2: AP token encoder
f.stage(8, 355, 760, 500, "Stage 2: Access-Point Token Encoder", "blue")
f.stacked_box(30, 425, 340, 150, "Per-AP CNN tokenizer<br>3 × Conv (3×5), pool 2×4<br>Linear (512 → 128)", "blue", n=4, fs=16)
f.text(30, 578, 160, 40, "shared over APs", fs=14)
f.circle(200, 650, 16, "⊕")
f.arrow([(200, 575), (200, 634)])
f.box(250, 620, 230, 60, "+ AP embedding " + it("e") + "<sub>" + it("a") + "</sub><br>+ Linear(" + it("p") + "<sub>" + it("t,a") + "</sub>)", "lightgreen", fs=15)
f.arrow([(250, 650), (216, 650)], head=False)
f.arrow([(200, 666), (200, 700)])
f.stacked_box(30, 700, 340, 70, "Set attention block × 2<br>(4 heads, pre-norm, FFN 256)", "teal", n=2, fs=16)
f.arrow([(370, 735), (430, 735)])
f.box(430, 700, 170, 70, "Attention<br>pooling", "darkorange", fs=16)
f.text(240, 800, 200, 40, "any number of APs", fs=14)
f.note(30, 795, 200, 50, "0.78 M parameters", fs=14)
f.text(610, 690, 150, 40, it("w") + "<sub>" + it("t,a") + "</sub> per AP", fs=15)
f.arrow([(600, 735), (660, 735), (660, 590), (780, 590), (780, 252), (830, 252)])
f.output_label(480, 780, 200, 40, it("z") + "<sub>" + it("t") + "</sub> ∈ " + R_("128"), fs=15)

# ------------------------------------------------------------------ Stage 3: memory sequence
f.stage(790, 10, 560, 520, "Stage 3: Channel Memory", "teal")
f.strip(815, 85, 8, cw=48, ch=44, colors=["#9fd0d0", "#8ccfcd", "#7bc4c2", "#6bb9b7", "#5fb3b1", "#53a7a5", "#479b99", "#3b8f8d"])
f.box(1205, 85, 60, 52, it("z") + "<sub>" + it("t") + "</sub>", "orange", fs=15)
f.text(815, 142, 390, 30, it("z") + "<sub>" + it("t−K") + "</sub> … " + it("z") + "<sub>" + it("t−1") + "</sub>: " + it("K") + " = 16 accepted snapshots", fs=14, align="left")
f.circle(1235, 175, 16, "⊕"); f.text(1255, 160, 90, 30, "query flag", fs=14, align="left")
f.arrow([(1235, 137), (1235, 159)])
f.box(830, 215, 490, 75, "Z = [" + it("z") + "<sub>" + it("t−K") + "</sub>, …, " + it("z") + "<sub>" + it("t−1") + "</sub>, " + it("z") + "<sub>" + it("t") + "</sub> + " + it("q") + "<sub>flag</sub>]", "lightteal", fs=17)
f.arrow([(1235, 191), (1235, 215)])
f.box(830, 330, 490, 70, "training: query = next snapshot (" + it("y") + " = 0)<br>or a snapshot " + it("d") + " m away (" + it("y") + " = 1)", "white", fs=15)
f.text(830, 410, 490, 40, "gaps 1–16, timing-offset and noise augmentation", fs=14)
f.arrow([(1070, 530), (1070, 572)])
f.output_label(1080, 532, 400, 40, it("Z") + " ∈ " + R_("(" + it("K") + "+1)×128"))

# ------------------------------------------------------------------ Stage 4: memory blocks
f.stage(790, 575, 560, 280, "Stage 4: Memory Blocks", "purple")
f.box(830, 640, 460, 80, "Memory block 1", "purple", fs=22)
f.arrow([(1070, 720), (1070, 745)])
f.stacked_box(830, 740, 460, 80, "Memory block 2", "purple", n=3, fs=22)
f.zoom_link((1290, 660), (1380, 90)); f.zoom_link((1290, 820), (1380, 760))
f.arrow([(1070, 855), (1070, 900)])
f.output_label(1080, 858, 400, 40, it("X̃") + " ∈ " + R_("(" + it("K") + "+1)×128"))

# ------------------------------------------------------------------ Detail panel
f.stage(1380, 10, 1170, 760, "Detailed Path-Encoded Forgetting Attention Block", "lavender")
f.box(1400, 80, 220, 60, it("x") + "<sub>" + it("s") + "</sub> (pre-norm)", "white", fs=16)
f.arrow([(1510, 140), (1510, 175)])
f.box(1400, 175, 240, 90, "Projections " + it("q") + ", " + it("k") + ", " + it("v") + "<br>(4 heads, " + it("d") + "<sub>h</sub> = 32)", "lightblue", fs=14)
f.box(1400, 290, 240, 110, "Householder direction<br>" + it("w") + "<sub>" + it("s") + "</sub>, ‖" + it("w") + "<sub>" + it("s") + "</sub>‖ = 1<br>" + it("β") + "<sub>" + it("s") + "</sub> = 2σ(" + it("a") + "<sup>⊤</sup>" + it("x") + "<sub>" + it("s") + "</sub>)", "orange", fs=14)
f.box(1400, 425, 240, 90, "Forget gate<br>" + it("f") + "<sub>" + it("s") + "</sub> = σ(" + it("w") + "<sub>" + it("f") + "</sub><sup>⊤</sup>" + it("x") + "<sub>" + it("s") + "</sub>)", "yellow", fs=14)
f.arrow([(1510, 265), (1510, 290)]); f.arrow([(1510, 400), (1510, 425)])
f.dashed_panel(1670, 80, 590, 435, title="Logit between query " + it("t") + " and slot " + it("s") + " ≤ " + it("t"))
f.box(1680, 135, 560, 70, it("H") + "<sub>" + it("s") + "</sub> = " + it("I") + " − " + it("β") + "<sub>" + it("s") + "</sub> " + it("w") + "<sub>" + it("s") + "</sub>" + it("w") + "<sub>" + it("s") + "</sub><sup>⊤</sup>", "orange", fs=17)
f.box(1680, 225, 560, 70, it("k̃") + "<sub>" + it("t,s") + "</sub> = " + it("H") + "<sub>" + it("t") + "</sub>" + it("H") + "<sub>" + it("t−1") + "</sub> ⋯ " + it("H") + "<sub>" + it("s+1") + "</sub> " + it("k") + "<sub>" + it("s") + "</sub>&nbsp;&nbsp;(PaTH)", "peach", fs=17)
f.box(1680, 315, 560, 80, it("a") + "<sub>" + it("t,s") + "</sub> = " + it("q") + "<sub>" + it("t") + "</sub><sup>⊤</sup>" + it("k̃") + "<sub>" + it("t,s") + "</sub> / √" + it("d") + "<sub>h</sub> + Σ<sub>" + it("r=s+1") + "</sub><sup>" + it("t") + "</sup> log " + it("f") + "<sub>" + it("r") + "</sub>", "lightblue", fs=17)
f.box(1680, 415, 560, 70, it("α") + "<sub>" + it("t,s") + "</sub> = softmax<sub>" + it("s≤t") + "</sub>(" + it("a") + "<sub>" + it("t,s") + "</sub>)&nbsp;&nbsp;(causal)", "yellow", fs=17)
f.arrow([(1960, 205), (1960, 225)]); f.arrow([(1960, 295), (1960, 315)]); f.arrow([(1960, 395), (1960, 415)])
f.arrow([(1640, 345), (1680, 345)], head=False); f.arrow([(1640, 470), (1660, 470), (1660, 355)], head=False)
f.arrow([(1640, 220), (1655, 220), (1655, 170), (1680, 170)], head=False)
f.text(1670, 525, 590, 40, "old keys are carried to the query along the path of channel changes", fs=15)
f.arrow([(1960, 485), (1960, 500), (2300, 500), (2300, 95)], head=False)
f.arrow([(2300, 95), (2330, 95)])
f.box(2330, 70, 200, 50, "Σ " + it("α") + "<sub>" + it("t,s") + "</sub> " + it("v") + "<sub>" + it("s") + "</sub>", "lightblue", fs=16)
f.arrow([(2430, 120), (2430, 150)])
f.box(2330, 150, 200, 50, "Output projection", "lightblue", fs=16)
f.arrow([(2430, 200), (2430, 230)])
f.box(2330, 230, 200, 50, "Residual + LayerNorm", "peach", fs=16)
f.arrow([(2430, 280), (2430, 310)])
f.box(2330, 310, 200, 80, "FFN<br>(128 → 256 → 128, GELU)", "lightblue", fs=16)
f.arrow([(2430, 390), (2430, 420)])
f.box(2330, 420, 200, 50, "Residual + LayerNorm", "peach", fs=16)
f.arrow([(2530, 255), (2548, 255), (2548, 445), (2530, 445)])
f.note(2330, 600, 210, 80, "ablations: RoPE / none,<br>no forget gate", fs=14)
f.note(1400, 560, 240, 70, it("O") + "(" + it("K") + "<sup>2</sup>" + it("d") + ") per identity<br>and SRS", fs=14)

# ------------------------------------------------------------------ Stage 5: head
f.stage(1380, 800, 760, 300, "Stage 5: Spoof Head", "orange")
f.box(1400, 870, 150, 90, "last two tokens<br>" + it("x̃") + "<sub>" + it("t−1") + "</sub>, " + it("x̃") + "<sub>" + it("t") + "</sub>", "darkorange", fs=15)
f.arrow([(1550, 915), (1575, 915)])
f.vbox(1575, 865, 55, 100, "Concat", "lavender", fs=15)
f.text(1400, 985, 240, 40, "[" + it("x̃") + "<sub>" + it("t") + "</sub> ; " + it("x̃") + "<sub>" + it("t") + "</sub> − " + it("x̃") + "<sub>" + it("t−1") + "</sub>]", fs=14)
f.arrow([(1630, 915), (1650, 915)])
f.vchain(1650, 855, 66, 130, [("Linear 256→128", "green"), ("GELU", "yellow"), ("Linear 128→1", "green"), ("Sigmoid", "red")], gap=20, fs=13)
f.arrow([(1994, 920), (2005, 920)])
f.text(2005, 885, 140, 70, "<b>Output:</b><br>spoof logit " + it("ℓ") + "<sub>" + it("t") + "</sub>", fs=16)
f.box(1650, 1020, 240, 50, "Prediction head " + it("ẑ") + "<sub>" + it("t") + "</sub> = " + it("W") + "<sub>p</sub>" + it("x̃") + "<sub>" + it("t−1") + "</sub>", "mauve", fs=14)
f.text(1895, 1010, 240, 70, "training: ℒ = BCE + 0.1 (1 − cos(" + it("ẑ") + "<sub>" + it("t") + "</sub>, " + it("z") + "<sub>" + it("t") + "</sub>))", fs=13, align="left")
f.arrow([(2135, 915), (2160, 915)])

# ------------------------------------------------------------------ Stage 6: response
f.stage(2160, 800, 390, 300, "Decision / Response", "pink", fs=20)
f.box(2180, 870, 170, 50, it("ℓ") + "<sub>" + it("t") + "</sub> &lt; " + it("τ") + ": accept, slide", "pink", fs=14)
f.box(2180, 930, 170, 50, it("ℓ") + "<sub>" + it("t") + "</sub> ≥ " + it("τ") + ": alarm, freeze", "pink", fs=14)
f.box(2180, 990, 170, 50, it("m") + " = 3 alarms: declare", "pink", fs=14)
f.arrow([(2350, 1015), (2375, 1015)])
f.box(2375, 870, 160, 50, "UPF: drop PDU", "mauve", fs=14)
f.box(2375, 930, 160, 50, "UDR: withdraw", "mauve", fs=14)
f.box(2375, 990, 160, 50, "AMF: reject re-reg.", "mauve", fs=14)
f.arrow([(2455, 990), (2455, 980)], head=False); f.arrow([(2455, 930), (2455, 920)], head=False)
f.text(2180, 1050, 355, 40, it("τ") + " = validation score at FPR 0.1 %", fs=14)

# loop back: memory update from decision to Stage 3
f.arrow([(2180, 895), (2160, 895), (2160, 1120), (1060, 1120), (1060, 905), (1055, 905)], dashed=True, head=False)
f.text(1100, 1090, 400, 40, "accepted " + it("z") + "<sub>" + it("t") + "</sub> joins the memory", fs=15, align="left")

path = os.path.join(OUT, "Figure1.drawio"); f.save(path, name="Figure1")
print("saved", path)
exe = r"C:\Program Files\draw.io\draw.io.exe"
if os.path.exists(exe):
    for fmt in ("png", "pdf"):
        subprocess.run([exe, "-x", "-f", fmt, "-s", "3" if fmt == "png" else "1", "-b", "10", "-o", os.path.join(OUT, f"Figure1.{fmt}"), path], check=False, timeout=180)
    print("exported png/pdf")
