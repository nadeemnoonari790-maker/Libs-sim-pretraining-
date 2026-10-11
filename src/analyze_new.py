import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, pickle, json, os
sys.path.insert(0, _SRC)
import numpy as np
from scipy.stats import wilcoxon, spearmanr
from data import *
RES = _R + "results/"
R = load_real(); Y = R["Ylog"]; G = R["groups"]; ug = np.unique(G)
TYPE = np.array(["alloy" if n.startswith(("C", "31X")) else ("ref" if n.startswith("J") else "natural") for n in R["names"]])
wl = np.load(CACHE + "real_blocks.npz")["wl"]
out = {}
# ---------------- MLP
mp = pickle.load(open(RES + "mlp_preds.pkl", "rb")); out["mlp"] = {}
def gerr(ms, k):
    e = np.mean([np.nanmean(np.abs(mp[f"{ms}|k{k}|s{s}"][:, SC_IDX] - Y[:, SC_IDX]), axis=1) for s in range(3)], axis=0)
    return np.array([np.nanmean(e[G == g]) for g in ug])
for k in (4, 8, 0):
    r = {}
    for m in ("scratch", "ft"):
        v = [metrics(mp[f"{m}|k{k}|s{s}"], Y)["macro_rmse"] for s in range(3)]; r[m] = (float(np.mean(v)), float(np.std(v)))
    d = gerr("ft", k) - gerr("scratch", k); r["p"] = float(wilcoxon(d[d != 0]).pvalue); r["frac_better"] = float(np.mean(d < 0))
    out["mlp"][k] = r; print("MLP", k, r)
# ---------------- cross-matrix
xm = pickle.load(open(RES + "xmat.pkl", "rb")); nat = TYPE == "natural"; out["xmat"] = {}
for setting, els in (("A_alloy+ref", SCORED), ("B_alloy", SCORED)):
    for method in ("scratch", "ft", "elperm"):
        ks = [k for k in xm if k.startswith(f"{setting}|{method}|")]
        if not ks: continue
        rm, sp = {e: [] for e in els}, {e: [] for e in els}
        for k in ks:
            P = xm[k]
            for e in els:
                j = TARGETS.index(e); v = nat & ~np.isnan(Y[:, j])
                rm[e].append(float(np.sqrt(np.mean((P[v, j] - Y[v, j]) ** 2)))); sp[e].append(float(spearmanr(P[v, j], Y[v, j]).statistic))
        out["xmat"][f"{setting}|{method}"] = dict(n=len(ks), rmse={e: float(np.mean(rm[e])) for e in els}, spearman={e: float(np.mean(sp[e])) for e in els},
                                                   macro_rmse=float(np.mean([np.mean(rm[e]) for e in els])), macro_spearman=float(np.mean([np.mean(sp[e]) for e in els])))
        print(setting, method, "n", len(ks), "macro RMSE %.3f macro rho %.2f" % (out["xmat"][f"{setting}|{method}"]["macro_rmse"], out["xmat"][f"{setting}|{method}"]["macro_spearman"]))
# paired site-group test ft vs scratch for xmat (natural samples)
def gerr_x(setting, m):
    e = np.mean([np.nanmean(np.abs(xm[f"{setting}|{m}|s{s}"][:, SC_IDX] - Y[:, SC_IDX]), axis=1) for s in range(3) if f"{setting}|{m}|s{s}" in xm], axis=0)
    gs = np.unique(G[nat]); return np.array([np.nanmean(e[nat & (G == g)]) for g in gs])
out["xmat_tests"] = {}
for setting in ("A_alloy+ref", "B_alloy"):
    for a_, b_ in (("ft", "scratch"), ("ft", "elperm")):
        try:
            d = gerr_x(setting, a_) - gerr_x(setting, b_); out["xmat_tests"][f"{setting}|{a_}_vs_{b_}"] = dict(median_diff=float(np.median(d)), frac_first_better=float(np.mean(d < 0)), p=float(wilcoxon(d[d != 0]).pvalue), n_groups=int(len(d)))
            print(setting, a_, "vs", b_, out["xmat_tests"][f"{setting}|{a_}_vs_{b_}"])
        except Exception as ex: print("skip", setting, a_, b_, ex)
# ---------------- saliency
if os.path.exists(RES + "sal.pkl"):
    sal = pickle.load(open(RES + "sal.pkl", "rb"))
    import sqlite3
    con = sqlite3.connect("file:" + _R + "data/atomic_lines.db?mode=ro", uri=True); W = {}
    for e in TARGETS:
        r = np.array(con.execute("select wavelength_nm,aki,gk,ek_ev from lines where element=? and sp_num in (1,2) and wavelength_nm between ? and ? and aki>0 and gk>0 and ek_ev<12", (e, wl[0], wl[-1])).fetchall(), float)
        w = r[:, 1] * r[:, 2] * np.exp(-r[:, 3] / 1.0); top = r[np.argsort(-w)[:12], 0]; m = np.zeros(NPIX, bool)
        for l in top: m |= np.abs(wl - l) <= 0.35
        W[e] = m
    def spec(S, e, j):
        """enrichment of attribution on e's own line windows relative to the other scored elements' windows (excluding overlaps)."""
        own = W[e]; oth = np.zeros(NPIX, bool)
        for e2 in SCORED:
            if e2 != e: oth |= W[e2]
        o1, o2 = own & ~oth, oth & ~own
        s = S[j] / S[j].sum()
        return (s[o1].sum() / o1.mean()) / (s[o2].sum() / o2.mean())
    out["sal"] = {}
    for m in ("scratch", "pretrained", "elperm", "pre_only"):
        per = {e: [] for e in SCORED}
        for key, d in sal.items():
            for e in SCORED: per[e].append(spec(d[m], e, TARGETS.index(e)))
        out["sal"][m] = dict(per_element={e: float(np.mean(v)) for e, v in per.items()}, macro=float(np.mean([np.mean(v) for v in per.values()])), n=len(sal))
        print("sal", m, "macro specificity %.2f" % out["sal"][m]["macro"], {e: round(float(np.mean(v)), 2) for e, v in per.items()})
    # paired (fold x seed) sign test pretrained vs scratch on per-element specificity, using 10 elements x n runs
    from scipy.stats import wilcoxon as wx
    def vec(m): return np.array([[spec(d[m], e, TARGETS.index(e)) for e in SCORED] for d in sal.values()]).mean(axis=0)
    for b_ in ("scratch", "elperm"):
        d = vec("pretrained") - vec(b_); out["sal"][f"pretrained_vs_{b_}"] = dict(frac_elements_higher=float(np.mean(d > 0)), p=float(wx(d).pvalue))
        print("pretrained vs", b_, out["sal"][f"pretrained_vs_{b_}"])
json.dump(out, open(RES + "new_analyses.json", "w"), indent=1, default=float)
