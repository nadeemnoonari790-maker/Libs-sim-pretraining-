import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, json, glob
sys.path.insert(0, _SRC)
import numpy as np, matplotlib.pyplot as plt, figstyle as fs
fs.use_ieee()
d = json.load(open(_R + "results/new_analyses.json"))
P = fs.PALETTE; col = {"scratch": "#8c8c8c", "ft": P[0], "elperm": P[1]}
lab = {"scratch": "Scratch", "ft": "Pre-trained", "elperm": "Element-permuted"}
fig, ax = plt.subplots(1, 3, figsize=(fs.TEXT_WIDTH_IN, 2.5), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1, 1]})
# (a) label-free elements, alloy-only training
LF = ["Ca", "Mg", "Mn", "Al", "Si", "Ba"]; w = 0.26; x = np.arange(len(LF))
for i, m in enumerate(("scratch", "ft", "elperm")):
    ax[0].bar(x + (i - 1) * w, [d["xmat"][f"B_alloy|{m}"]["spearman"][e] for e in LF], w, color=col[m], label=lab[m])
ax[0].axhline(0, color="k", lw=0.5); ax[0].set_xticks(x); ax[0].set_xticklabels(LF)
ax[0].set_ylabel("Spearman ρ on natural samples"); ax[0].set_ylim(-0.4, 1.15); ax[0].legend(frameon=False, fontsize=6, loc="upper center", ncol=3, columnspacing=0.8, handlelength=1.0, bbox_to_anchor=(0.55, 1.0))
# (b) attribution specificity
ms = ["scratch", "elperm", "pretrained"]; nm = ["Scratch", "Element-\npermuted", "Pre-trained"]
SC = ["Cu", "Zn", "Fe", "Pb", "Ca", "Mg", "Mn", "Al", "Si", "Ba"]
for i, m in enumerate(ms):
    v = [d["sal"][m]["per_element"][e] for e in SC]
    ax[1].bar(i, d["sal"][m]["macro"], 0.6, color={"scratch": col["scratch"], "elperm": col["elperm"], "pretrained": col["ft"]}[m], alpha=0.85)
    ax[1].scatter(np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), v, s=5, color="k", zorder=3, lw=0)
ax[1].axhline(1, color="k", lw=0.6, ls=":"); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(nm, fontsize=6.5)
ax[1].set_ylabel("Own-line attribution\nspecificity (×)")
# (c) MLP
ks = ["4", "8", "0"]; x = np.arange(3)
for i, m in enumerate(("scratch", "ft")):
    ax[2].bar(x + (i - 0.5) * 0.36, [d["mlp"][k][m][0] for k in ks], 0.36, yerr=[d["mlp"][k][m][1] for k in ks], color=col[m], capsize=1.5, error_kw={"lw": 0.6}, label=lab[m])
ax[2].set_xticks(x); ax[2].set_xticklabels(["4", "8", "all"]); ax[2].set_xlabel("Labelled samples"); ax[2].set_ylabel("Macro RMSE (log$_{10}$ wt%)")
ax[2].set_ylim(0.5, 1.0); ax[2].legend(frameon=False, fontsize=6)
for a_, l in zip(ax, "abc"): fs.add_panel_label(a_, f"({l})")
fs.save(fig, _R + "figures/fig8_new_analyses", formats=("pdf", "svg", "png"))
