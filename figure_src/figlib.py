"""
figlib.py — helpers for generating "stage-flow" architecture figures as
editable draw.io (mxGraph) XML. Self-contained; copy next to your generator
script and `from figlib import Fig`.

Coordinates: design in a virtual canvas (e.g. 2560×1300 like a high-res figure)
and let `scale` map it to page units, OR set scale=1 and design directly.

    from figlib import Fig
    f = Fig(page_w=1560, page_h=800, scale=0.6)
    f.stage(8, 10, 980, 260, "Stage 1: Input Preprocessing", "green")
    a = f.box(15, 75, 345, 75, "Raw Features<br>(76 dims)", "gray")
    f.arrow([(360,112),(383,112),(383,155),(405,155)])
    f.save("/mnt/user-data/outputs/figure.drawio")
"""
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET

FONT = "fontFamily=Helvetica;"

# ------------------------------------------------------------------ palette
# Stage containers: (fill, gradient) — pastel bands with bold black titles.
STAGE = {
    "green":    ("#cfe6c6", "#e6f3df"),
    "blue":     ("#bcd8ee", "#d6e8f6"),
    "teal":     ("#a9d6d4", "#c9e6e5"),
    "purple":   ("#a89ed6", "#c3bbe6"),
    "lavender": ("#e6dbee", "#f2ebf6"),   # zoom-in / detail panels
    "orange":   ("#f7d0a3", "#fbe3c6"),
    "pink":     ("#f0b0dd", "#f7cfea"),   # explainability / evaluation
    "yellow":   ("#f6ebb8", "#fbf4d8"),
    "gray":     ("#dedede", "#efefef"),
}
# Inner boxes: (fill, stroke) — role-based, reuse consistently inside a figure.
BOX = {
    "gray":       ("#e6e6e6", "#999999"),   # raw inputs / neutral
    "lightgreen": ("#d9ead0", "#8fb08a"),   # normalization / linear (green)
    "green":      ("#cfe6c6", "#7fa676"),
    "blue":       ("#79ade0", "#3b6ea8"),   # main encoder layers
    "lightblue":  ("#c9d7ee", "#6b86b8"),   # projection / FFN / residual path
    "teal":       ("#8ccfcd", "#2f8a88"),
    "lightteal":  ("#d3ecec", "#2f8a88"),
    "purple":     ("#8a7ec8", "#4e4295"),   # sequence / SSM / attention blocks
    "lavender":   ("#c3b8e6", "#7d6bb8"),
    "orange":     ("#f3b48c", "#c07a4a"),   # conv / gating
    "darkorange": ("#f4a25a", "#c97a2a"),   # pooling
    "peach":      ("#f6c68a", "#c98b3a"),   # residual + norm
    "yellow":     ("#f9e5a0", "#c9b35a"),   # activations / dropout
    "mauve":      ("#d8c7d8", "#8a6d8a"),
    "red":        ("#f2a3a3", "#c05a5a"),   # softmax / output
    "pink":       ("#e08ad2", "#a04a92"),   # XAI
    "white":      ("#ffffff", "#444444"),
    "panel":      ("#e9f3fb", "#444444"),   # small inset panels (plots)
}
STRIP_COLORS = ["#7fa8d8", "#f2b27a", "#c9c0d0", "#b9bfe6", "#a8d8b0", "#f4a6a6"]


class Fig:
    def __init__(self, page_w=1560, page_h=800, scale=1.0, name="Figure"):
        self.cells, self._id = [], 1
        self.page_w, self.page_h, self.s, self.name = page_w, page_h, scale, name

    # ---------------------------------------------------------- primitives
    def _nid(self):
        self._id += 1
        return f"n{self._id}"

    def _sc(self, v):
        return round(v * self.s)

    def vertex(self, value, style, x, y, w, h):
        i = self._nid()
        v = escape(value, {'"': "&quot;"})
        self.cells.append(
            f'<mxCell id="{i}" value="{v}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{self._sc(x)}" y="{self._sc(y)}" width="{self._sc(w)}" '
            f'height="{self._sc(h)}" as="geometry"/></mxCell>')
        return i

    def edge(self, points, style):
        i = self._nid()
        pts = [(self._sc(x), self._sc(y)) for x, y in points]
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        mid = ""
        if len(pts) > 2:
            arr = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in pts[1:-1])
            mid = f'<Array as="points">{arr}</Array>'
        self.cells.append(
            f'<mxCell id="{i}" style="{style}" edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
            f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/>{mid}</mxGeometry></mxCell>')
        return i

    # ---------------------------------------------------------- components
    def rect(self, x, y, w, h, label="", fill="#ffffff", stroke="#333333", fs=16,
             bold=False, grad=None, sw=1.5, arc=12, extra="", color="#000000"):
        st = (f"rounded=1;arcSize={arc};whiteSpace=wrap;html=1;fillColor={fill};"
              f"strokeColor={stroke};strokeWidth={sw};fontSize={fs};fontColor={color};" + FONT + extra)
        if grad:
            st += f"gradientColor={grad};gradientDirection=south;"
        if bold:
            st += "fontStyle=1;"
        return self.vertex(label, st, x, y, w, h)

    def text(self, x, y, w, h, label, fs=16, bold=False, align="center",
             valign="middle", extra="", color="#000000"):
        st = (f"text;html=1;align={align};verticalAlign={valign};fontSize={fs};"
              f"whiteSpace=wrap;fontColor={color};" + FONT + extra)
        if bold:
            st += "fontStyle=1;"
        return self.vertex(label, st, x, y, w, h)

    def stage(self, x, y, w, h, title, color="blue", fs=24, title_h=50, title_w=None,
              stroke="#333333"):
        """Pastel gradient stage container with a bold black title at top-left."""
        fill, grad = STAGE[color] if color in STAGE else (color, None)
        c = self.rect(x, y, w, h, "", fill, stroke, grad=grad, arc=6)
        self.text(x + 15, y + 8, title_w or (w - 30), title_h, title, fs=fs,
                  bold=True, align="left")
        return c

    def box(self, x, y, w, h, label, role="white", fs=16, bold=False, arc=12,
            extra="", color="#000000"):
        fill, stroke = BOX[role] if role in BOX else (role, "#333333")
        return self.rect(x, y, w, h, label, fill, stroke, fs=fs, bold=bold,
                         arc=arc, extra=extra, color=color)

    def stacked_box(self, x, y, w, h, label, role="blue", n=3, off=9, fs=17, **kw):
        """Box with n-1 shadow copies behind it (repeated layers / batches)."""
        fill, stroke = BOX[role] if role in BOX else (role, "#333333")
        for k in range(n - 1, 0, -1):
            darker = _shade(fill, -0.07 * k)
            self.rect(x + off * k, y - off * k * 0.5, w, h, "", darker, stroke, arc=12)
        return self.rect(x, y, w, h, label, fill, stroke, fs=fs, **kw)

    def vbox(self, x, y, w, h, label, role="lavender", fs=15):
        """Tall narrow box with vertical (bottom-to-top) text — for head chains."""
        return self.box(x, y, w, h, label, role, fs=fs, extra="horizontal=0;", arc=10)

    def vchain(self, x0, y, w, h, items, gap=25, fs=15):
        """Horizontal chain of vbox items [(label, role), ...] joined by arrows."""
        ids = []
        for i, (lab, role) in enumerate(items):
            x = x0 + i * (w + gap)
            ids.append(self.vbox(x, y, w, h, lab, role, fs))
            if i < len(items) - 1:
                self.arrow([(x + w, y + h / 2), (x + w + gap - 2, y + h / 2)])
        return ids, x0 + len(items) * (w + gap) - gap

    def dashed_panel(self, x, y, w, h, title=None, fill="#ede4f3", fs=16):
        p = self.rect(x, y, w, h, "", fill, "#333333", arc=8, extra="dashed=1;", sw=1.2)
        if title:
            self.text(x + 5, y + 5, w - 10, 40, title, fs=fs, bold=True)
        return p

    def circle(self, cx, cy, r, label="", fill="#ffffff", stroke="#000000", fs=16):
        return self.vertex(label, f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};"
                                  f"strokeColor={stroke};fontSize={fs};fontStyle=1;" + FONT,
                           cx - r, cy - r, 2 * r, 2 * r)

    def strip(self, x, y, n, cw=36, ch=34, colors=None):
        """Row of colored cells (token / embedding vector glyph)."""
        colors = colors or STRIP_COLORS
        for k in range(n):
            self.rect(x + k * cw, y, cw, ch, "", colors[k % len(colors)], "#666666",
                      arc=0, sw=1)
        return x + n * cw

    def curve_panel(self, x, y, w, h, title=None, curves=None):
        """Inset plot: box + axes + smooth curves (learnable functions, splines)."""
        self.box(x, y, w, h, "", "panel", arc=8)
        if title:
            self.text(x + 10, y + 5, w - 20, 45, title, fs=17)
        top = y + (55 if title else 15)
        mid = (top + y + h - 15) / 2
        self.arrow([(x + 10, mid), (x + w - 10, mid)])
        self.arrow([(x + w / 2, y + h - 15), (x + w / 2, top)])
        curves = curves or [("#e8963c", 2.5), ("#5f7fa8", 2.0), ("#8fa8c4", 2.0)]
        import math
        for j, (col, sw) in enumerate(curves):
            pts = []
            for k in range(7):
                px = x + 15 + (w - 30) * k / 6
                py = mid + (h * 0.18) * math.sin(k * 1.3 + j * 0.9) * (1 + 0.15 * j)
                pts.append((px, py))
            self.arrow(pts, head=False, curved=True, color=col, sw=sw)

    def note(self, x, y, w, h, label, fs=17):
        """Plain gray side note (parameter count, complexity, dataset size)."""
        return self.box(x, y, w, h, label, "#ebebeb", fs=fs)

    # ---------------------------------------------------------- arrows
    def arrow(self, pts, head=True, dashed=False, sw=1.5, color="#000000", curved=False):
        st = (f"html=1;rounded=0;strokeColor={color};strokeWidth={sw};"
              f"endArrow={'block' if head else 'none'};endFill=1;" + FONT)
        if dashed:
            st += "dashed=1;"
        if curved:
            st += "curved=1;"
        return self.edge(pts, st)

    def zoom_link(self, p_from, p_to):
        """Thin dashed line connecting a block to its zoom-in detail panel."""
        return self.arrow([p_from, p_to], head=False, dashed=True, sw=1.2)

    def output_label(self, x, y, w, h, body, fs=17, align="left"):
        """'Output: ...' annotation next to an inter-stage arrow."""
        return self.text(x, y, w, h, f"<b>Output:</b> {body}", fs=fs, align=align)

    # ---------------------------------------------------------- save
    def to_xml(self):
        inner = "\n        ".join(self.cells)
        return f'''<mxfile host="app.diagrams.net" agent="Claude" version="24.0.0" type="device">
  <diagram id="fig" name="{escape(self.name)}">
    <mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.page_w}" pageHeight="{self.page_h}" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {inner}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''

    def save(self, path, name=None):
        if name:
            self.name = name
        xml = self.to_xml()
        ET.fromstring(xml)  # raises on malformed XML
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(xml)
        print(f"saved {path}: {len(self.cells)} cells, XML valid")
        return path


def _shade(hexcol, amt):
    """Darken (amt<0) or lighten (amt>0) a #rrggbb color."""
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    f = lambda c: max(0, min(255, int(c + 255 * amt)))
    return f"#{f(r):02x}{f(g):02x}{f(b):02x}"


# Math / symbol helpers for HTML labels
def R_(body):        # ℝ^{...}
    return f"ℝ<sup>{body}</sup>"

def it(s):           # italic
    return f"<i>{s}</i>"

def sub(base, s):    # X_sub
    return f"{base}<sub>{s}</sub>"

SYM = dict(arrow="→", elem="∈", odot="⊙", otimes="⊗", oplus="⊕", sigma="σ",
           phi="φ", approx="≈", times="×", prime="′")
