"""Figure 1 (draw.io) - CloneGuard methodology drawn in the block-diagram style of the user's reference figures:
white background, dashed layer frames with red titles (Input Layer / Memory Encoder Layer xN / Output Layer),
tensors as cubes and coloured row strips with (shape) labels, attention matrix as a grid, (x)/(+) operators,
and a zoom-in (dashed leaders) of the novel path-encoded key module.  python gen_fig1.py"""
import os, subprocess
from figlib import Fig, FONT
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "figures", "drawio"); os.makedirs(OUT, exist_ok=True)
f = Fig(page_w=1500, page_h=820, scale=0.6, name="Figure1")

RED, INK, GREY = "#d40000", "#000000", "#666666"
BLUE, BLUE_D, TEAL, TEAL_D, GREEN, GREEN_D, ORANGE, ORANGE_D, LILAC, LILAC_D, YEL, YEL_D = \
    "#dae8fc", "#6c8ebf", "#b0e3e6", "#0e8088", "#d5e8d4", "#82b366", "#ffe6cc", "#d79b00", "#e1d5e7", "#9673a6", "#fff2cc", "#d6b656"


def box(x, y, w, h, label, fill=BLUE, stroke=BLUE_D, fs=15, bold=False, arc=8, dashed=False, sw=1.2):
    st = (f"rounded=1;arcSize={arc};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontSize={fs};" + FONT)
    if bold: st += "fontStyle=1;"
    if dashed: st += "dashed=1;"
    return f.vertex(label, st, x, y, w, h)


def frame(x, y, w, h, title, fill="#ffffff", stroke="#000000", tw=360):
    f.vertex("", f"rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=1.2;dashed=1;" + FONT, x, y, w, h)
    f.text(x + 10, y + 4, tw, 34, title, fs=18, bold=True, color=RED, align="left")


def cube(x, y, w, h, label, fill, stroke, fs=14, depth=14):
    st = (f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.08;darkOpacity2=0.16;size={depth};"
          f"fillColor={fill};strokeColor={stroke};fontSize={fs};" + FONT)
    return f.vertex(label, st, x, y, w, h)


def rows(x, y, w, n, rh, colors, gap=4):
    for i in range(n):
        f.vertex("", f"rounded=1;arcSize=30;whiteSpace=wrap;html=1;fillColor={colors[i % len(colors)]};strokeColor=none;" + FONT, x, y + i * (rh + gap), w, rh)


def grid(x, y, n, cell, fills):
    for i in range(n):
        for j in range(n):
            fc = fills(i, j)
            f.vertex("", f"whiteSpace=wrap;html=1;fillColor={fc};strokeColor=#9aa5b1;strokeWidth=0.6;" + FONT, x + j * cell, y + i * cell, cell, cell)


def op(cx, cy, sym, r=15):
    return f.vertex(sym, "ellipse;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1.2;fontSize=18;" + FONT, cx - r, cy - r, 2 * r, 2 * r)


def A(pts, head=True, dashed=False, sw=1.3, color="#000000"):
    f.arrow(pts, head=head, dashed=dashed, sw=sw, color=color)


def lab(x, y, w, h, s, fs=14, align="center", color=INK, bold=False):
    f.text(x, y, w, h, s, fs=fs, align=align, color=color, bold=bold)


it = lambda s: f"<i>{s}</i>"

# =============================================================== Input layer (left column, bottom -> top)
frame(10, 10, 560, 1280, "Input Layer", tw=200)
# raw snapshots: a 3-D stack of channel matrices (A x M rows, F columns), K+1 of them
lab(30, 1254, 520, 30, "Input: uplink channel snapshots of one identity (" + it("K") + "+1 = 17: history + query)", fs=12)
SHADES = ["#f2f2f2", "#dfe8f3", "#c6d8ec", "#a9c5e2", "#8db2d8", "#6f9fcf", "#4f8bc4", "#2f77b9", "#1f5f9e"]
for k in range(5):
    x0, y0 = 50 + k * 22, 1225 - k * 16
    f.vertex("", f"whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#4f6b8a;strokeWidth=0.9;" + FONT, x0, y0 - 120, 190, 120)
    for i in range(5):
        for j in range(8):
            f.vertex("", f"whiteSpace=wrap;html=1;fillColor={SHADES[(i * 3 + j * 2 + k) % len(SHADES)]};strokeColor=none;" + FONT, x0 + 5 + j * 22.5, y0 - 115 + i * 22, 21, 20)
lab(350, 1090, 200, 60, it("H") + "<sub>" + it("t") + "</sub> ∈ ℂ<sup>" + it("A") + "×" + it("M") + "×" + it("F") + "</sup><br>rows: AP × antenna<br>columns: sub-band", fs=12, align="left")
lab(40, 1228, 60, 20, it("t") + "−" + it("K"), fs=11); lab(230, 1150, 60, 20, it("t"), fs=11)
A([(180, 1041), (180, 990)])
box(80, 930, 200, 60, "Normalisation<br>(unit energy, AP phase ref.)", GREEN, GREEN_D, fs=12)
A([(180, 930), (180, 880)])
box(80, 810, 200, 70, "Per-AP CNN tokenizer<br>Conv(3×5) ×3, Linear", BLUE, BLUE_D, fs=12)
cube(310, 815, 130, 60, it("U") + " (" + it("K") + "+1, " + it("A") + ", " + it("d") + ")", "#f8cecc", "#b85450", fs=12)
A([(280, 845), (310, 845)])
box(330, 710, 210, 50, "AP embedding " + it("e") + "<sub>" + it("a") + "</sub> ⊕ power " + it("p") + "<sub>" + it("t,a") + "</sub>", YEL, YEL_D, fs=12)
A([(375, 815), (375, 760)], head=False)
A([(180, 810), (180, 700)])
box(80, 630, 200, 70, "Set self-attention<br>over APs ×2 (4 heads)", LILAC, LILAC_D, fs=12)
A([(330, 735), (280, 665)], head=False)
A([(180, 630), (180, 590)])
box(80, 530, 200, 60, "Attention pooling", ORANGE, ORANGE_D, fs=13)
A([(180, 530), (180, 480)])
# token embedding rows (K+1 rows, gradient) = the memory sequence
lab(30, 110, 520, 30, "Memory sequence " + it("Z") + " (" + it("K") + "+1, " + it("d") + ")", fs=14, bold=True)
ROWC = ["#fde9d9", "#fbd5bb", "#f9c19c", "#f6ad7e", "#f39a60", "#f08642", "#e97226", "#d85f12", "#c04f0c", "#a84208"]
for i in range(10):
    rows(60, 150 + i * 30, 240, 1, 24, [ROWC[i]])
lab(300, 150, 40, 24, it("z") + "<sub>" + it("t−K") + "</sub>", fs=11, align="left"); lab(300, 420, 60, 24, it("z") + "<sub>" + it("t") + "</sub> + " + it("q") + "<sub>flag</sub>", fs=11, align="left")
lab(14, 448, 160, 36, "query flag on<br>the last row", fs=10, color=GREY)
box(365, 172, 190, 62, "accepted snapshots<br>slide in, oldest out", "#ffffff", GREY, fs=11, dashed=True)
A([(180, 480), (180, 455)])
A([(300, 290), (330, 290), (330, 203), (365, 203)], head=False, dashed=True, sw=1.0, color=GREY)
# from memory sequence to the encoder layer
A([(300, 300), (610, 300)])
lab(310, 270, 290, 26, "(" + it("n") + ", " + it("d") + "),  " + it("n") + " = " + it("K") + "+1", fs=13)

# =============================================================== Memory Encoder Layer xN (centre)
frame(600, 10, 1430, 740, "Memory Encoder Layer ×2 (path-encoded forgetting attention)", fill="#fbfbfd", tw=900)
# three parallel projections
for i, (nm, yy, col, cs) in enumerate((("Linear", 150, BLUE, BLUE_D), ("Linear", 300, TEAL, TEAL_D), ("Linear", 450, GREEN, GREEN_D))):
    A([(610, 300), (650, 300), (650, yy + 25), (690, yy + 25)] if i != 1 else [(610, 300), (690, 325)])
    box(690, yy, 110, 50, nm, col, cs, fs=14)
cube(830, 130, 150, 70, it("Q") + " (" + it("n") + ", " + it("d") + "<sub>h</sub>)", "#dae8fc", "#6c8ebf")
cube(830, 280, 150, 70, it("K") + " (" + it("n") + ", " + it("d") + "<sub>h</sub>)", "#b0e3e6", "#0e8088")
cube(830, 430, 150, 70, it("V") + " (" + it("n") + ", " + it("d") + "<sub>h</sub>)", "#d5e8d4", "#82b366")
for yy in (175, 325, 475): A([(800, yy), (830, yy)])
# gate + householder projections (small)
box(690, 560, 110, 40, "Linear", YEL, YEL_D, fs=13); A([(610, 300), (650, 300), (650, 580), (690, 580)], head=False)
box(690, 620, 110, 40, "Linear", ORANGE, ORANGE_D, fs=13); A([(650, 580), (650, 640), (690, 640)], head=False)
lab(810, 560, 230, 40, "forget gate " + it("f") + "<sub>" + it("s") + "</sub> = σ(" + it("w") + "<sub>f</sub><sup>⊤</sup>" + it("x") + "<sub>" + it("s") + "</sub>)", fs=12, align="left")
lab(810, 620, 230, 40, it("w") + "<sub>" + it("s") + "</sub>, " + it("β") + "<sub>" + it("s") + "</sub> = 2σ(" + it("a") + "<sup>⊤</sup>" + it("x") + "<sub>" + it("s") + "</sub>)", fs=12, align="left")
# path-encoded key module (novel block)
box(1040, 250, 230, 130, "Path-encoded<br>key module<br>(Householder product)", "#ffffff", "#000000", fs=15, bold=True)
A([(980, 315), (1040, 315)])
A([(1040, 640), (1130, 640), (1130, 380)], head=False)
cube(1300, 280, 150, 70, it("K̃") + " (" + it("n") + ", " + it("n") + ", " + it("d") + "<sub>h</sub>)", "#b0e3e6", "#0e8088")
A([(1270, 315), (1300, 315)])
# Q x K~ -> logits + forget -> softmax -> attention matrix
op(1510, 230, "⊗"); A([(980, 165), (1510, 165), (1510, 215)]); A([(1450, 315), (1510, 315), (1510, 245)], head=False)
op(1510, 560, "⊕"); A([(1040, 580), (1495, 580), (1495, 560)], head=False)
lab(1200, 505, 300, 30, "+ Σ<sub>" + it("r") + "=" + it("s") + "+1</sub><sup>" + it("t") + "</sup> log " + it("f") + "<sub>" + it("r") + "</sub>  (cumulative forget)", fs=12)
A([(1510, 245), (1510, 545)], head=False)
box(1570, 540, 120, 40, "Softmax", "#ffffff", "#000000", fs=14); A([(1525, 560), (1570, 560)])
lab(1540, 180, 200, 60, it("a") + "<sub>" + it("t,s") + "</sub> = " + it("q") + "<sub>" + it("t") + "</sub><sup>⊤</sup>" + it("k̃") + "<sub>" + it("t,s") + "</sub>/√" + it("d") + "<sub>h</sub>", fs=13, align="left")
# attention matrix grid (causal, lower-triangular)
GX, GY, NC, CS = 1720, 390, 8, 26
grid(GX, GY, NC, CS, lambda i, j: ("#ffffff" if j > i else ["#f3f7fb", "#dce8f5", "#c3d7ee", "#a5c3e4", "#86afda", "#6a9bd0", "#4e86c4", "#3570b4"][min(7, (i - j) // 1)]))
lab(GX - 20, GY + NC * CS + 2, NC * CS + 40, 24, "attention (" + it("n") + ", " + it("n") + "), causal", fs=12)
A([(1690, 560), (1720, 560)])
# attention x V -> output
op(1824, 690, "⊗"); A([(1824, 598), (1824, 675)]); A([(980, 465), (1010, 465), (1010, 690), (1809, 690)])
cube(1870, 655, 150, 70, "Output (" + it("n") + ", " + it("d") + ")", "#fff2cc", "#d6b656"); A([(1839, 690), (1870, 690)])
# right column: add&norm, FFN, add&norm (as in the reference)
box(2060, 150, 170, 44, "Concat heads", LILAC, LILAC_D, fs=13); A([(2020, 690), (2145, 690), (2145, 194)], head=False)
box(2060, 220, 170, 44, "Linear", BLUE, BLUE_D, fs=13); A([(2145, 194), (2145, 220)])
box(2060, 290, 170, 44, "Add &amp; Norm", ORANGE, ORANGE_D, fs=13); A([(2145, 264), (2145, 290)])
box(2060, 360, 170, 44, "Feed forward", GREEN, GREEN_D, fs=13); A([(2145, 334), (2145, 360)])
box(2060, 430, 170, 44, "Add &amp; Norm", ORANGE, ORANGE_D, fs=13); A([(2145, 404), (2145, 430)])
A([(2230, 312), (2255, 312), (2255, 452), (2230, 452)])     # residual
A([(610, 300), (620, 300), (620, 70), (2270, 70), (2270, 312), (2230, 312)], head=False, dashed=True, sw=1.0, color=GREY)   # residual from input
A([(2145, 474), (2145, 760)]); lab(2155, 490, 140, 60, it("X̃") + " ∈ ℝ<sup>" + it("n") + "×" + it("d") + "</sup>", fs=14, align="left")

# =============================================================== zoom-in: path-encoded key module (bottom centre)
frame(600, 780, 1430, 510, "Path-encoded Key Module (PaTH + forget gate)", fill="#f4f6fb", tw=700)
A([(1040, 380), (1052, 780)], head=False, dashed=True, sw=1.0); A([(1270, 380), (1900, 780)], head=False, dashed=True, sw=1.0)
cube(640, 880, 150, 70, it("K") + " (" + it("n") + ", " + it("d") + "<sub>h</sub>)", "#b0e3e6", "#0e8088")
cube(640, 1000, 150, 70, it("w") + "<sub>" + it("s") + "</sub>, " + it("β") + "<sub>" + it("s") + "</sub> (" + it("n") + ", " + it("d") + "<sub>h</sub>)", "#ffe6cc", "#d79b00", fs=12)
A([(790, 1035), (840, 1035)])
box(840, 1005, 260, 60, it("H") + "<sub>" + it("s") + "</sub> = " + it("I") + " − " + it("β") + "<sub>" + it("s") + "</sub>" + it("w") + "<sub>" + it("s") + "</sub>" + it("w") + "<sub>" + it("s") + "</sub><sup>⊤</sup>", ORANGE, ORANGE_D, fs=15)
lab(840, 1070, 260, 26, "one reflection per snapshot", fs=12, color=GREY)
# chain of H's (cubes) illustrating the accumulated product along the path
for i, lbl in enumerate(("H<sub>s+1</sub>", "H<sub>s+2</sub>", "⋯", "H<sub>t</sub>")):
    x0 = 1160 + i * 120
    if lbl == "⋯":
        lab(x0, 1015, 100, 40, "⋯", fs=22)
    else:
        cube(x0, 1000, 100, 60, "<i>" + lbl + "</i>", "#ffe6cc", "#d79b00", fs=13)
    if i < 3: A([(x0 + 100, 1030), (x0 + 120, 1030)], head=False)
A([(1100, 1035), (1160, 1035)], head=False)
A([(790, 915), (900, 915), (900, 1005)], head=False)
op(1680, 1030, "⊗"); A([(1640, 1030), (1665, 1030)]); A([(900, 915), (1680, 915), (1680, 1015)], head=False)
A([(1695, 1030), (1740, 1030)])
cube(1740, 995, 220, 70, it("k̃") + "<sub>" + it("t,s") + "</sub> = " + it("H") + "<sub>" + it("t") + "</sub>⋯" + it("H") + "<sub>" + it("s") + "+1</sub>" + it("k") + "<sub>" + it("s") + "</sub>", "#b0e3e6", "#0e8088", fs=12)
lab(640, 1110, 1370, 30, "old keys are transported to the query through the reflections of every snapshot in between: the logit follows the channel path, not elapsed time", fs=13, color=GREY)
box(640, 1170, 480, 50, "ablations: rotary positions / no positions / no forget gate", "#ffffff", GREY, fs=12, dashed=True)
lab(1180, 1170, 560, 50, it("O") + "(" + it("n") + "<sup>2</sup>" + it("d") + ") per identity and SRS; 0.78 M parameters", fs=13, color=GREY)

# =============================================================== Output layer (right column)
frame(2050, 780, 530, 510, "Output Layer", tw=200)
box(2080, 840, 200, 50, "last two tokens<br>" + it("x̃") + "<sub>" + it("t−1") + "</sub>, " + it("x̃") + "<sub>" + it("t") + "</sub>", BLUE, BLUE_D, fs=13)
A([(2145, 760), (2145, 840)])
box(2080, 910, 200, 44, "Concat [" + it("x̃") + "<sub>" + it("t") + "</sub>; Δ" + it("x̃") + "]", LILAC, LILAC_D, fs=13); A([(2180, 890), (2180, 910)])
box(2080, 975, 200, 44, "FCN (256 → 128 → 1)", GREEN, GREEN_D, fs=12); A([(2180, 954), (2180, 975)])
box(2080, 1040, 200, 44, "Sigmoid", ORANGE, ORANGE_D, fs=13); A([(2180, 1019), (2180, 1040)])
A([(2180, 1084), (2180, 1120)])
f.vertex("", "ellipse;whiteSpace=wrap;html=1;fillColor=#d40000;strokeColor=#d40000;" + FONT, 2164, 1120, 32, 32)
lab(2200, 1110, 200, 50, "Spoof logit " + it("ℓ") + "<sub>" + it("t") + "</sub>", fs=15, color=RED, bold=True, align="left")
box(2320, 840, 240, 50, "Prediction head<br>" + it("ẑ") + "<sub>" + it("t") + "</sub> = " + it("W") + "<sub>p</sub>" + it("x̃") + "<sub>" + it("t−1") + "</sub> (training)", "#ffffff", GREY, fs=12, dashed=True)
A([(2280, 865), (2320, 865)], head=False, dashed=True, sw=1.0, color=GREY)
box(2080, 1170, 480, 56, "owner stream: memory slides, " + it("m") + " = 3 alarms → declare<br>new stream of a known identity: probation, " + it("P") + " = 3", "#ffffff", "#000000", fs=11)
box(2080, 1236, 480, 44, "core response: UPF drops the PDU · UDR withdraws the subscription · AMF rejects re-registration", "#fde9d9", "#b85450", fs=11)
A([(2180, 1152), (2180, 1170)])

path = os.path.join(OUT, "Figure1.drawio"); f.save(path, name="Figure1")
print("saved", path)
exe = r"C:\Program Files\draw.io\draw.io.exe"
if os.path.exists(exe):
    for fmt in ("png", "pdf"):
        subprocess.run([exe, "-x", "-f", fmt, "-s", "3" if fmt == "png" else "1", "-b", "10", "-o", os.path.join(OUT, f"Figure1.{fmt}"), path], check=False, timeout=180)
    print("exported png/pdf")
