"""Figure for dataset 2 and the cross-instrument test (drawn at text width, IEEE style).
(a) ChemLIBS: macro RMSE vs labelled targets; (b) relative gain of pre-training over scratch on both datasets;
(c) deep-sea: simulation vs real-cross-instrument pre-training; (d) ChemLIBS attribution specificity."""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, json, glob
sys.path.insert(0, _SRC)
import numpy as np, matplotlib.pyplot as plt, figstyle as fs
fs.use_ieee()
RES = _R + "results/"
S2 = json.load(open(RES + "summary2.json")); S1 = json.load(open(RES + "summary.json"))
T2 = S2["table"]; T1 = S1["table"]
P = fs.PALETTE
C = {"null": "#bdbdbd", "pls": P[2], "cnn_real": "#6e6e6e", "cnn_ft": P[0], "cnn_elperm": P[1], "cnn_mix": P[3],
     "cnn_xreal": P[4], "cnn_xrealperm": P[5] if len(P) > 5 else "#a6761d"}
L = {"null": "Predict mean", "pls": "PLS", "cnn_real": "CNN scratch", "cnn_ft": "Sim.-pretrained", "cnn_elperm": "Element-permuted",
     "cnn_mix": "Real+synthetic mix", "cnn_xreal": "Pretrained on ChemLIBS (real)", "cnn_xrealperm": "ChemLIBS, permuted labels"}

fig, ax = plt.subplots(2, 2, figsize=(fs.TEXT_WIDTH_IN, 4.6), layout="constrained")
# (a) dataset 2 label budget
K2 = [4, 8, 16, 32, 128, 512, 0]; K2 = [k for k in K2 if f"cnn_ft|k{k}" in T2]
x = np.arange(len(K2)); a = ax[0, 0]
for m in ("null", "pls", "cnn_real", "cnn_elperm", "cnn_mix", "cnn_ft"):
    ks = [k for k in K2 if f"{m}|k{k}" in T2]
    if not ks:
        continue
    xi = [K2.index(k) for k in ks]; mu = np.array([T2[f"{m}|k{k}"]["macro_rmse"] for k in ks]); sd = np.array([T2[f"{m}|k{k}"]["sd_over_seeds"] for k in ks])
    a.plot(xi, mu, "-o", ms=2.5, lw=1.0 if m != "cnn_ft" else 1.5, color=C[m], label=L[m], ls="--" if m == "cnn_mix" else "-")
    a.fill_between(xi, mu - sd, mu + sd, color=C[m], alpha=0.15, lw=0)
a.set_xticks(x); a.set_xticklabels([str(k) if k else "all" for k in K2]); a.set_xlabel("Labelled training targets")
a.set_ylabel("Macro RMSE (log$_{10}$ wt%)"); a.legend(frameon=False, fontsize=6, ncol=2, loc="upper right")
fs.add_panel_label(a, "(a) ChemLIBS", loc="lower left")
# (b) relative gain both datasets
a = ax[0, 1]
for T, ks, lab, c, mk in ((T1, [4, 8, 16, 32, 0], "Deep-sea (68 samples)", P[1], "s"), (T2, K2, "ChemLIBS (2,645 targets)", P[0], "o")):
    xs = [K2.index(k) if k in K2 else None for k in ks]
    g = [100 * (T[f"cnn_real|k{k}"]["macro_rmse"] - T[f"cnn_ft|k{k}"]["macro_rmse"]) / T[f"cnn_real|k{k}"]["macro_rmse"] for k in ks]
    if T is T1:     # dataset-1 element-permuted control lives in extra.json (k = 4, 8, all)
        EM = json.load(open(RES + "extra.json"))["macro"]
        ge = [100 * (EM["scratch"][str(k)][0] - EM["elem_permuted"][str(k)][0]) / EM["scratch"][str(k)][0]
              if str(k) in EM["elem_permuted"] else np.nan for k in ks]
    else:
        ge = [100 * (T[f"cnn_real|k{k}"]["macro_rmse"] - T[f"cnn_elperm|k{k}"]["macro_rmse"]) / T[f"cnn_real|k{k}"]["macro_rmse"]
              if f"cnn_elperm|k{k}" in T else np.nan for k in ks]
    # connect only consecutive budgets that exist for this dataset; 'all' is drawn as a separate point if there is a gap
    xs = np.array(xs, float); g = np.array(g); ge = np.array(ge)
    seg = [i for i in range(len(ks)) if ks[i] != 0]; last = [i for i in range(len(ks)) if ks[i] == 0]
    contiguous = len(seg) and xs[seg[-1]] + 1 == xs[last[0]]
    idx_line = seg + (last if contiguous else [])
    a.plot(xs[idx_line], g[idx_line], "-" + mk, color=c, ms=3, lw=1.3, label=lab + ", correct labels")
    if not contiguous:
        a.plot(xs[last], g[last], mk, color=c, ms=3)
    if not np.all(np.isnan(ge)):
        ok = [i for i in idx_line if not np.isnan(ge[i])]
        a.plot(xs[ok], ge[ok], ":" + mk, color=c, ms=3, lw=1.0, mfc="white", label=lab + ", element-permuted")
        if not contiguous and not np.isnan(ge[last[0]]):
            a.plot(xs[last], ge[last], mk, color=c, ms=3, mfc="white")
a.axhline(0, color="k", lw=0.5); a.set_xticks(x); a.set_xticklabels([str(k) if k else "all" for k in K2])
a.set_xlabel("Labelled training samples"); a.set_ylabel("RMSE reduction vs scratch (%)"); a.legend(frameon=False, fontsize=5.6, loc="lower left"); a.set_ylim(top=28)
fs.add_panel_label(a, "(b)", loc="upper left")
# (c) cross-instrument on deep-sea
a = ax[1, 0]; XI = S2.get("xinst", {}); K1 = ["4", "8", "16", "32", "0"]
for m in ("cnn_real", "cnn_xrealperm", "cnn_xreal", "cnn_ft"):
    ks = [k for k in K1 if k in XI and m in XI[k]]
    if not ks:
        continue
    xi = [K1.index(k) for k in ks]; mu = np.array([XI[k][m]["mean"] for k in ks]); sd = np.array([XI[k][m]["sd"] for k in ks])
    a.plot(xi, mu, "-o", ms=2.5, lw=1.0 if m != "cnn_ft" else 1.5, color=C[m], label=L[m])
    a.fill_between(xi, mu - sd, mu + sd, color=C[m], alpha=0.15, lw=0)
a.set_xticks(range(5)); a.set_xticklabels(["4", "8", "16", "32", "all"]); a.set_xlabel("Labelled deep-sea training samples")
a.set_ylabel("Macro RMSE (log$_{10}$ wt%)"); a.legend(frameon=False, fontsize=6, loc="upper right")
fs.add_panel_label(a, "(c) Deep-sea target", loc="lower left")
# (d) attribution specificity dataset 2
a = ax[1, 1]
if "sal2" in S2:
    sd_ = S2["sal2"]; ms = ["scratch", "elperm", "pretrained", "pre_only"]; nm = ["Scratch", "Element-\npermuted", "Pre-trained", "Synthetic\nonly"]
    cc = {"scratch": C["cnn_real"], "elperm": C["cnn_elperm"], "pretrained": C["cnn_ft"], "pre_only": "#9ecae1"}
    for i, m in enumerate(ms):
        v = [x_ for x_ in sd_[m]["per_element"].values() if x_ == x_]
        a.bar(i, sd_[m]["macro"], 0.6, color=cc[m], alpha=0.85)
        a.scatter(np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), v, s=5, color="k", zorder=3, lw=0)
    a.axhline(1, color="k", lw=0.6, ls=":"); a.set_xticks(range(4)); a.set_xticklabels(nm, fontsize=6.5)
    a.set_ylabel("Own-line attribution\nspecificity (×)")
fs.add_panel_label(a, "(d) ChemLIBS", loc="upper left")
print(fs.save(fig, _R + "figures/fig9_dataset2", formats=("pdf", "svg", "png")))
