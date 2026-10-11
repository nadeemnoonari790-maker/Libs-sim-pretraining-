"""Minimal plotting style used for the paper figures (serif fonts, colour-blind-safe palette, de-boxed axes,
figures saved at their true size as PDF + SVG + 600-dpi PNG)."""
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

COL_WIDTH_IN, TEXT_WIDTH_IN, WIDE_WIDTH_IN = 3.5, 7.16, 5.0
FS_BASE, FS_TICK, FS_LEGEND, FS_SMALL = 8, 7, 7, 6
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]  # Okabe-Ito
GREY_DARK, GREY_MID, GREY_LIGHT = "#4D4D4D", "#999999", "#D9D9D9"
ACCENT = "#C8102E"


def use_ieee(base_fontsize=FS_BASE):
    have = {f.name for f in font_manager.fontManager.ttflist}
    chain = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"]
    serif = [c for c in chain if c in have] + [c for c in chain if c not in have]
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": serif, "font.size": base_fontsize, "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND,
        "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333", "axes.prop_cycle": matplotlib.cycler(color=PALETTE), "xtick.direction": "out",
        "ytick.direction": "out", "xtick.major.size": 3.0, "ytick.major.size": 3.0, "axes.grid": False,
        "grid.color": GREY_LIGHT, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
        "legend.handlelength": 1.4, "legend.labelspacing": 0.3, "lines.linewidth": 1.4, "lines.markersize": 4.0,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "standard", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none"})
    return serif


def figure(width_in=COL_WIDTH_IN, height_in=None, aspect=0.66, **kw):
    return plt.subplots(figsize=(width_in, height_in or width_in * aspect), layout="constrained", **kw)


def save(fig, path_noext, formats=("pdf", "svg", "png"), dpi=600, close=True):
    os.makedirs(os.path.dirname(os.path.abspath(path_noext)), exist_ok=True)
    out = []
    for ext in formats:
        fig.savefig(f"{path_noext}.{ext}", dpi=dpi)
        out.append(f"{path_noext}.{ext}")
    if close:
        plt.close(fig)
    return out


def add_panel_label(ax, text, loc="upper left", pad=0.02):
    x, y, ha, va = {"upper left": (pad, 1 - pad, "left", "top"), "upper right": (1 - pad, 1 - pad, "right", "top"),
                    "lower left": (pad, pad, "left", "bottom"), "lower right": (1 - pad, pad, "right", "bottom")}[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=FS_BASE, fontweight="bold", color="#1A1A1A")
