"""Schematic figures: (1) method overview, (2) 1D-CNN architecture. One spec -> vector PDF/SVG/PNG (matplotlib) and an
editable .drawio file each.
NOTE: needs the block-diagram helpers arch_mpl / arch_drawio (not included); the rendered figures and the editable
.drawio files are provided in figures/ (open .drawio in https://app.diagrams.net)."""
import sys, glob, copy
import os
sys.path.insert(0, os.environ.get("ARCH_HELPERS", "."))
import figstyle as fs
import arch_mpl as am
import arch_drawio as ad
fs.use_ieee()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures") + "/"
B, E = ad.block, ad.edge
PAD = 30
FS = 7.0      # in-figure text size (pt) for the matplotlib render


def both(blocks, edges, name, width_in=fs.TEXT_WIDTH_IN, extra=None):
    mb = [dict(b, fontsize=b.get("fs", FS)) for b in blocks]
    fig, ax = am.render(mb, edges, width_in=width_in, pad=PAD)
    if extra:
        extra(ax)
    paths = fs.save(fig, OUT + name, formats=("pdf", "svg", "png"))
    db = [dict(b, fontsize=11) for b in blocks]
    ad.write_drawio(db, edges, OUT + name + ".drawio")
    print(paths)


# --------------------------------------------------------------------------- Figure: method overview
ov = [
    # simulation branch (top row)
    B("db", "Atomic line list\\n(NIST-derived)", 0, 0, 105, 46, role="input", shape="parallelogram"),
    B("fm", "LTE slab forward model\\nSaha–Boltzmann, radiative\\ntransfer, Stark widths", 125, -4, 168, 54, role="encoder"),
    B("dr", "Domain randomisation\\ncomposition · plasma ·\\ninstrument · noise", 313, -4, 145, 54, role="encoder"),
    B("sy", "Synthetic spectra\\n24,000 per seed\\n+ exact labels", 478, -2, 122, 50, role="input", shape="parallelogram"),
    B("pt", "Pre-train 1D-CNN\\n(4,000 steps)", 625, 0, 120, 46, role="bottleneck"),
    # real-data branch (middle row)
    B("rd", "Real spectra\\ndeep-sea ChemiCam (68)\\nChemLIBS rocks (2,645)", -20, 110, 165, 54, role="input", shape="parallelogram"),
    B("fo", "Group-held-out folds\\n(sites / suites)", 160, 114, 125, 46, role="neutral"),
    B("kb", "k labelled samples\\nk = 4 … all", 310, 114, 120, 46, role="head"),
    B("ft", "Fine-tune\\n(500 steps)", 625, 114, 120, 46, role="bottleneck"),
    B("ev", "Test on held-out groups\\nmacro RMSE · bootstrap CI\\npaired Wilcoxon", 775, 108, 170, 58, role="output"),
    # controls (bottom row)
    B("ct", "Label controls: sample-shuffled,\\nelement-permuted synthetic labels", 434, -85, 210, 46, role="loss", dashed=True),
    B("bl", "Baselines: scratch CNN, PLS,\\nmean, real+synthetic mix", 160, 205, 190, 46, role="neutral", dashed=True),
    B("xi", "Cross-instrument: pre-train on\\nreal ChemLIBS spectra", 610, 205, 190, 46, role="attention", dashed=True),
]
ov_e = [
    E("db", "fm"), E("fm", "dr"), E("dr", "sy"), E("sy", "pt"),
    E("rd", "fo"), E("fo", "kb"), E("kb", "ft"),
    E("pt", "ft", label="weights"), E("ft", "ev"),
    E("ct", "sy", dashed=True, exit="S", entry="N"),
    E("bl", "kb", dashed=True, exit="N", entry="S"),
    E("xi", "ft", dashed=True, exit="N", entry="S"),
]
for e in ov_e:
    e["label"] = e["label"].replace("\\n", "\n")

# --------------------------------------------------------------------------- Figure: CNN architecture
W, H = 100, 52
cnn = [
    B("in", "Input spectrum\\n1 × 1024\\n(L2-normalised)", 0, 0, W, H, role="input", shape="parallelogram"),
    B("c1", "Conv1d 9, s2\\nBN · GELU\\n16 × 512", 120, 0, W, H, role="encoder"),
    B("c2", "Conv1d 7, s2\\nBN · GELU\\n32 × 256", 240, 0, W, H, role="encoder"),
    B("c3", "Conv1d 5, s2\\nBN · GELU\\n64 × 128", 360, 0, W, H, role="encoder"),
    B("c4", "Conv1d 5, s2\\nBN · GELU\\n96 × 64", 480, 0, W, H, role="encoder"),
    B("ap", "AvgPool 2\\n96 × 32", 600, 0, W, H, role="encoder"),
    B("fl", "Flatten 3072\\nDropout 0.2", 600, 96, W, H, role="head"),
    B("f1", "Linear 128\\nGELU\\nDropout 0.1", 480, 96, W, H, role="head"),
    B("f2", "Linear n\\n(n = 11 or 10\\nelements)", 360, 96, W, H, role="head"),
    B("out", "log₁₀ wt%\\nper element", 240, 96, W, H, role="output", shape="parallelogram"),
    B("ls", "Masked Huber loss\\n(missing labels\\nignored)", 120, 96, W, H, role="loss"),
]
cnn_e = [E("in", "c1"), E("c1", "c2"), E("c2", "c3"), E("c3", "c4"), E("c4", "ap"),
         E("ap", "fl", exit="S", entry="N"), E("fl", "f1"), E("f1", "f2"), E("f2", "out"), E("out", "ls")]


def cnn_extra(ax):
    # brackets naming the transferable feature extractor and the regression head
    y = -16
    ax.plot([120, 700], [y, y], color="#6C8EBF", lw=1.0)
    for x in (120, 700):
        ax.plot([x, x], [y, y + 6], color="#6C8EBF", lw=1.0)
    ax.text(410, y - 4, "Feature extractor f (frozen in the head-only control)", ha="center", va="bottom", fontsize=FS, color="#2F4F7F")
    y2 = 160
    ax.plot([360, 700], [y2, y2], color="#B8962E", lw=1.0)
    for x in (360, 700):
        ax.plot([x, x], [y2, y2 - 6], color="#B8962E", lw=1.0)
    ax.text(530, y2 + 4, "Regression head h", ha="center", va="top", fontsize=FS, color="#7A6420")


if __name__ == "__main__":
    both(ov, ov_e, "fig0_overview")
    both(cnn, cnn_e, "fig0_cnn", extra=cnn_extra)
