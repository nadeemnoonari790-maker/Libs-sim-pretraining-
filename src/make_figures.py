"""Publication figures (IEEE-style helpers, drawn at true column width). Usage: python3 -I src/make_figures.py fig2 fig3 ..."""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, json, pickle, glob
sys.path.insert(0, _SRC)
SK = _SRC
sys.path.insert(0, SK)
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs
from data import *

fs.use_ieee()
OUT = _R + "figures/"
R = load_real()
WL = np.load(CACHE + "real_blocks.npz")["wl"]
NS = len(R["names"])


def fig_spectra():
    """(a) real brass vs best-fit forward model (certified composition, fitted nuisance parameters);
    (b) real basalt with the unmodelled detector-edge bands marked; (c) a typical synthetic draw."""
    import fit_alloys as FA
    th = json.load(open(_R + "results/expA_fit_fig2.json"))["C2600"]
    sim = FA.simulate(np.array(th["theta"]), FA.COMP["C2600"])
    s = R["names"].index("C2600"); real_b = R["Y"][R["start"][s]:R["start"][s] + R["count"][s]].mean(0)
    s2 = R["names"].index("JB-2"); real_r = R["Y"][R["start"][s2]:R["start"][s2] + R["count"][s2]].mean(0)
    z = np.load(CACHE + "synth_informed_s1.npz"); k = 17
    syn = z["Y"][k]
    r_fit = float(np.corrcoef(real_b[(WL >= 330) & (WL <= 545)], sim[(WL >= 330) & (WL <= 545)])[0, 1])
    fig, axes = plt.subplots(3, 1, figsize=(fs.TEXT_WIDTH_IN, 4.6), sharex=True, layout="constrained")
    ax = axes[0]
    ax.plot(WL, real_b / real_b.max(), color=fs.GREY_DARK, lw=0.7, label="Real")
    ax.plot(WL, -sim / sim.max(), color=fs.PALETTE[1], lw=0.7, label="Forward model, fitted nuisance parameters")
    ax.axhline(0, color=fs.GREY_MID, lw=0.4); ax.set_yticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticklabels(["1", "0.5", "0", "0.5", "1"])
    fs.add_panel_label(ax, "(a) Brass C2600", loc="upper left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.50, 1.0), frameon=False, ncol=2)
    for ax, y, lab, c in [(axes[1], real_r, "(b) Real basalt JB-2", fs.GREY_DARK), (axes[2], syn, "(c) Synthetic spectrum (random draw)", fs.PALETTE[1])]:
        ax.plot(WL, y / y.max(), color=c, lw=0.7)
        fs.add_panel_label(ax, lab, loc="upper left" if ax is axes[2] else "upper right")
    for ax in axes[1:]:
        for lo, hi in [(WL[0], 333.0), (545.0, WL[-1])]:
            ax.axvspan(lo, hi, color=fs.GREY_LIGHT, alpha=0.55, lw=0)
    axes[1].text(553.0, 0.62, "edge\nbands", ha="center", fontsize=fs.FS_SMALL, color=fs.GREY_DARK)
    for ax in axes:
        ax.set_ylabel("norm. intensity"); ax.set_xlim(WL[0], WL[-1])
    axes[-1].set_xlabel("Wavelength (nm)")
    info = dict(brass_fit_pearson_330_545=r_fit)
    json.dump(info, open(_R + "results/fig2_info.json", "w"), indent=1)
    print(fs.save(fig, OUT + "fig2_spectra", formats=("pdf", "svg", "png")), info)


# ---------------------------------------------------------------- result figures
def _load(path):
    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        return {}


MAIN = None
KS = [4, 8, 16, 32, 0]
KLAB = ["4", "8", "16", "32", "all\n(~54)"]


def _series(res, method, k):
    """list of (seed, pred[NS,NT]) available."""
    return [(int(key.split("|")[2][1:]), v) for key, v in res.items() if key.startswith(f"{method}|k{k}|")]


def _rmse_stats(res, method, k):
    v = [metrics(p, R["Ylog"])["macro_rmse"] for _, p in _series(res, method, k)]
    return (np.mean(v), np.std(v), len(v)) if v else (np.nan, np.nan, 0)


def fig_curve():
    res = _load(_R + "results/main_preds.pkl"); ctrl = _load(_R + "results/ctrl_shuf.pkl")
    fig, ax = fs.figure(width_in=fs.COL_WIDTH_IN, aspect=0.82)
    spec = [("null", "Predict mean", fs.GREY_LIGHT, "--", None, res), ("pls", "PLS", fs.GREY_MID, "-", "s", res),
            ("cnn_real", "CNN, real labels only", fs.GREY_DARK, "-", "o", res),
            ("cnn_ft", "CNN, synthetic pre-training", fs.ACCENT, "-", "o", res),
            ("cnn_mix", "CNN, mixed real/synthetic", fs.PALETTE[1], "-.", "v", res),
            ("cnn_ft", "CNN, shuffled-label pre-training", fs.PALETTE[3], ":", "^", ctrl)]
    x = np.arange(len(KS))
    for m, lab, c, ls, mk, src in spec:
        st = [_rmse_stats(src, m, k) for k in KS]
        mu = np.array([t[0] for t in st]); sd = np.array([t[1] for t in st]); ok = ~np.isnan(mu)
        if not ok.any():
            continue
        ax.plot(x[ok], mu[ok], color=c, ls=ls, lw=1.6 if c == fs.ACCENT else 1.1, marker=mk, ms=3.2, mec="white", mew=0.5, label=lab, zorder=3)
        if m != "null" and np.nanmax(sd[ok]) > 0:
            ax.fill_between(x[ok], (mu - sd)[ok], (mu + sd)[ok], color=c, alpha=0.18, lw=0, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(KLAB)
    ax.set_xlabel("Labelled real samples used for training")
    ax.set_ylabel("Macro RMSE (log$_{10}$ wt%)")
    ax.yaxis.grid(True); ax.set_ylim(0.5, 1.15)
    ax.legend(loc="lower left", frameon=False, fontsize=fs.FS_SMALL + 0.5)
    print(fs.save(fig, OUT + "fig3_label_budget"))


def fig_elements(k=8):
    res = _load(_R + "results/main_preds.pkl")
    names = [("null", "Predict mean", fs.GREY_LIGHT), ("cnn_real", "CNN, real only", fs.GREY_DARK), ("cnn_ft", "CNN, synthetic pre-training", fs.ACCENT)]
    fig, ax = fs.figure(width_in=fs.TEXT_WIDTH_IN, aspect=0.32)
    w = 0.26
    for i, (m, lab, c) in enumerate(names):
        vals = []
        for e in SCORED:
            v = [metrics(p, R["Ylog"])[e] for _, p in _series(res, m, k)]
            vals.append(np.mean(v) if v else np.nan)
        ax.bar(np.arange(len(SCORED)) + (i - 1) * w, vals, w, color=c, label=lab, zorder=3, lw=0)
    ax.set_xticks(np.arange(len(SCORED))); ax.set_xticklabels(SCORED)
    ax.set_ylabel("RMSE (log$_{10}$ wt%)"); ax.yaxis.grid(True)
    ax.legend(loc="upper right", frameon=False, ncol=3)
    print(fs.save(fig, OUT + f"fig4_elements_k{k}"))


TYPE = lambda n: "alloy" if n.startswith(("C", "31X")) else ("ref" if n.startswith("J") else "natural")


def fig_parity(k=8):
    res = _load(_R + "results/main_preds.pkl")
    els = ["Cu", "Zn", "Fe", "Pb"]
    fig, axes = plt.subplots(2, 4, figsize=(fs.TEXT_WIDTH_IN, 3.7), layout="constrained", sharex=True, sharey=True)
    col = {"alloy": fs.PALETTE[0], "ref": fs.PALETTE[2], "natural": fs.PALETTE[4]}
    lab = {"alloy": "Alloy standards", "ref": "GSJ reference rocks", "natural": "Seafloor / natural samples"}
    for r, (m, title) in enumerate([("cnn_real", "CNN, real only"), ("cnn_ft", "CNN, synthetic pre-training")]):
        sr = _series(res, m, k)
        if not sr:
            continue
        P = np.mean([p for _, p in sr], axis=0)
        for c, e in enumerate(els):
            j = TARGETS.index(e); ax = axes[r, c]
            y = R["Ylog"][:, j]; v = ~np.isnan(y)
            for t in col:
                idx = np.array([TYPE(n) == t for n in R["names"]]) & v
                ax.scatter(y[idx], P[idx, j], s=7, color=col[t], alpha=0.85, lw=0, label=lab[t] if (r == 0 and c == 0) else None, zorder=3)
            ax.plot([-2.2, 2.2], [-2.2, 2.2], color=fs.GREY_MID, lw=0.6, zorder=1)
            ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.3, 2.3)
            if r == 0:
                ax.set_title(e, fontsize=fs.FS_BASE)
            if c == 0:
                ax.set_ylabel(f"{title}\npredicted log$_{{10}}$ wt%")
            if r == 1:
                ax.set_xlabel("certified log$_{10}$ wt%")
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=fs.FS_SMALL)
    print(fs.save(fig, OUT + f"fig5_parity_k{k}"))


def fig_ablation():
    """(a) fine-tuning at k=4/8 after pre-training on different synthetic sets (seed mean ± s.d.); (b) masked edge bands (robustness, seed 0)."""
    base = _load(_R + "results/main_preds.pkl"); ctrl = _load(_R + "results/ctrl_shuf.pkl")
    abl = {t: _load(_R + f"results/abl_{t}.pkl") for t in ["broad", "nonoise", "noambient"]}
    crop = _load(_R + "results/crop_preds.pkl")
    conds = [("No pre-training", base, "cnn_real", fs.GREY_DARK), ("Default synthetic", base, "cnn_ft", fs.ACCENT),
             ("Broad instrument prior", abl["broad"], "cnn_ft", fs.PALETTE[0]), ("No noise model", abl["nonoise"], "cnn_ft", fs.PALETTE[2]),
             ("No seawater/nuisance lines", abl["noambient"], "cnn_ft", fs.PALETTE[4]), ("Shuffled labels", ctrl, "cnn_ft", fs.PALETTE[3])]
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT_WIDTH_IN, 2.7), layout="constrained", gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]; w = 0.13
    for i, (lab, src, m, c) in enumerate(conds):
        vals, sds = [], []
        for k in (4, 8):
            t = _rmse_stats(src, m, k); vals.append(t[0]); sds.append(t[1])
        ax.bar(np.arange(2) + (i - 2.5) * w, vals, w, yerr=sds, color=c, label=lab, zorder=3, lw=0, error_kw=dict(lw=0.6, capsize=1.2, zorder=4))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["4 labelled samples", "8 labelled samples"])
    ax.set_ylabel("Macro RMSE (log$_{10}$ wt%)"); ax.set_ylim(0.5, 1.05); ax.yaxis.grid(True)
    ax.legend(loc="upper right", frameon=False, fontsize=fs.FS_SMALL, ncol=2)
    fs.add_panel_label(ax, "(a)", loc="upper left")
    ax = axes[1]; w = 0.32
    for i, (lab, m, c) in enumerate([("Real only", "cnn_real", fs.GREY_DARK), ("Synthetic pre-training", "cnn_ft", fs.ACCENT)]):
        vals = []
        for k in (4, 8, 0):
            v = [metrics(p, R["Ylog"])["macro_rmse"] for s_, p in _series(crop, m, k) if s_ == 0]
            vals.append(v[0] if v else np.nan)
        ax.bar(np.arange(3) + (i - 0.5) * w, vals, w, color=c, label=lab, zorder=3, lw=0)
    ax.set_xticks(range(3)); ax.set_xticklabels(["4", "8", "all"]); ax.set_xlabel("Labelled real samples (edge bands masked)")
    ax.set_ylim(0.5, 1.05); ax.yaxis.grid(True); ax.legend(loc="upper right", frameon=False, fontsize=fs.FS_SMALL)
    fs.add_panel_label(ax, "(b)", loc="upper left")
    print(fs.save(fig, OUT + "fig6_ablations"))


def fig_mechanism():
    """(a) What carries the benefit: scratch / shuffled / element-permuted labels / frozen extractor / full fine-tuning (3 seeds).
    (b) Pre-training set-size scaling (seed 0)."""
    main = _load(_R + "results/main_preds.pkl"); head = _load(_R + "results/head_preds.pkl")
    elp = _load(_R + "results/ctrl_elperm.pkl"); shuf = _load(_R + "results/ctrl_shuf.pkl")
    conds = [("Scratch (no pre-training)", main, "cnn_real", fs.GREY_DARK), ("Shuffled labels", shuf, "cnn_ft", fs.PALETTE[3]),
             ("Element-permuted labels", elp, "cnn_ft", fs.PALETTE[4]), ("Frozen extractor, head only", head, "cnn_head", fs.PALETTE[0]),
             ("Full fine-tuning", main, "cnn_ft", fs.ACCENT)]
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT_WIDTH_IN, 2.8), layout="constrained", gridspec_kw=dict(width_ratios=[1.7, 1]))
    ax = axes[0]; w = 0.16
    for i, (lab, src, m, c) in enumerate(conds):
        mu, sd = [], []
        for k in (4, 8, 0):
            t = _rmse_stats(src, m, k); mu.append(t[0]); sd.append(t[1])
        ax.bar(np.arange(3) + (i - 2) * w, mu, w, yerr=sd, color=c, label=lab, zorder=3, lw=0, error_kw=dict(lw=0.6, capsize=1.2, zorder=4))
    ax.set_xticks(range(3)); ax.set_xticklabels(["4 labelled samples", "8 labelled samples", "all labels (~54)"])
    ax.set_ylabel("Macro RMSE (log$_{10}$ wt%)"); ax.set_ylim(0.5, 1.12); ax.yaxis.grid(True)
    ax.legend(loc="upper right", frameon=False, fontsize=fs.FS_SMALL, ncol=2); fs.add_panel_label(ax, "(a)", loc="upper left")
    ax = axes[1]
    sc = {}
    for n, f, m in [(1500, "scale_n1500", "cnn_ft"), (6000, "scale_n6000", "cnn_ft"), (24000, "main_preds", "cnn_ft"), (96000, "scale_n96000", "cnn_ft")]:
        src = _load(_R + f"results/{f}.pkl")
        for k in (4, 8):
            v = [metrics(p, R["Ylog"])["macro_rmse"] for s_, p in _series(src, m, k) if s_ == 0]
            if v: sc.setdefault(k, {})[n] = v[0]
    for k, c, lab in [(4, fs.ACCENT, "4 labelled samples"), (8, fs.PALETTE[0], "8 labelled samples")]:
        if k in sc:
            xs = sorted(sc[k]); ax.plot(range(len(xs)), [sc[k][n] for n in xs], color=c, marker="o", ms=3.2, mec="white", mew=0.5, lw=1.3, label=lab, zorder=3)
            base = [metrics(p, R["Ylog"])["macro_rmse"] for s_, p in _series(main, "cnn_real", k) if s_ == 0][0]
            ax.axhline(base, color=c, ls=":", lw=0.9, zorder=2)
    ax.set_xticks(range(4)); ax.set_xticklabels(["1.5k", "6k", "24k", "96k"]); ax.set_xlabel("Synthetic spectra in pre-training set")
    ax.set_ylabel("Macro RMSE (log$_{10}$ wt%)"); ax.yaxis.grid(True); ax.legend(loc="center right", frameon=False, fontsize=fs.FS_SMALL)
    ax.text(0.03, 0.89, "dotted lines: no pre-training", transform=ax.transAxes, fontsize=fs.FS_SMALL, va="top", color=fs.GREY_DARK)
    fs.add_panel_label(ax, "(b)", loc="lower left")
    print(fs.save(fig, OUT + "fig7_mechanism"))


if __name__ == "__main__":
    for a in sys.argv[1:]:
        {"fig2": fig_spectra, "fig3": fig_curve, "fig4": fig_elements, "fig5": fig_parity, "fig6": fig_ablation, "fig7": fig_mechanism}.get(a, lambda: print("unknown", a))()
