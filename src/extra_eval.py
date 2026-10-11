"""Aggregates the extra experiments: head-only fine-tuning, element-permuted control, pre-training set-size scaling, AdaBN probe.
Group-level paired Wilcoxon tests use per-site-group mean |error| (seed-averaged), like evaluate.py."""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, pickle, json, os
sys.path.insert(0, _SRC)
import numpy as np
from scipy.stats import wilcoxon
from data import *
R = load_real(); Y = R["Ylog"]; G = R["groups"]; ug = np.unique(G)
L = lambda f: pickle.load(open(_R + f"results/{f}.pkl", "rb")) if os.path.exists(_R + f"results/{f}.pkl") else {}
main, head, elp, shuf = L("main_preds"), L("head_preds"), L("ctrl_elperm"), L("ctrl_shuf")

def preds(src, m, k): return [(int(key.split("|")[2][1:]), v) for key, v in src.items() if key.startswith(f"{m}|k{k}|")]
def macro(src, m, k):
    v = [metrics(p, Y)["macro_rmse"] for _, p in preds(src, m, k)]; return float(np.mean(v)), float(np.std(v)), len(v)
def gerr(src, m, k):
    e = np.mean([np.nanmean(np.abs(p[:, SC_IDX] - Y[:, SC_IDX]), axis=1) for _, p in preds(src, m, k)], axis=0)
    return np.array([np.nanmean(e[G == g]) for g in ug])
def test(a, b):
    d = gerr(*a) - gerr(*b); nz = d[d != 0]
    return dict(median_diff=float(np.median(d)), frac_first_better=float(np.mean(d < 0)), p=float(wilcoxon(nz).pvalue))

conds = {"scratch": (main, "cnn_real"), "full_ft": (main, "cnn_ft"), "head_only": (head, "cnn_head"), "elem_permuted": (elp, "cnn_ft"), "shuffled": (shuf, "cnn_ft")}
out = {"macro": {}, "tests": {}}
print(f"{'':16}" + "".join(f"{'k='+str(k):>16}" for k in (4, 8, 0)))
for c, (src, m) in conds.items():
    row = {k: macro(src, m, k) for k in (4, 8, 0)}; out["macro"][c] = row
    print(f"{c:16}" + "".join(f"{row[k][0]:.3f}±{row[k][1]:.3f} n{row[k][2]}  " for k in (4, 8, 0)))
for k in (4, 8, 0):
    for a, b in [("full_ft", "scratch"), ("head_only", "scratch"), ("head_only", "full_ft"), ("elem_permuted", "scratch"), ("full_ft", "elem_permuted"), ("elem_permuted", "shuffled")]:
        t = test((conds[a][0], conds[a][1], k), (conds[b][0], conds[b][1], k)); out["tests"][f"{a}_vs_{b}|k{k}"] = t
        print(f"  {a} vs {b} k={k}: d={t['median_diff']:+.3f} first-better {100*t['frac_first_better']:.0f}% p={t['p']:.2g}")
# scaling (seed 0)
sc = {}
for n, src, m in [(1500, L("scale_n1500"), "cnn_ft"), (6000, L("scale_n6000"), "cnn_ft"), (24000, main, "cnn_ft"), (96000, L("scale_n96000"), "cnn_ft")]:
    r = {}
    for k in (4, 8):
        v = [metrics(p, Y)["macro_rmse"] for s_, p in preds(src, m, k) if s_ == 0]
        if v: r[k] = v[0]
    if r: sc[n] = r
out["scaling_seed0"] = sc; print("scaling (seed 0):", sc)
sc0 = {k: metrics([p for s_, p in preds(main, "cnn_real", k) if s_ == 0][0], Y)["macro_rmse"] for k in (4, 8)}; print("scratch seed0:", sc0)
json.dump(out, open(_R + "results/extra.json", "w"), indent=1, default=float)

# ---- generator ablations (seed mean ± sd over available seeds) + tests against scratch and against the default generator
abl = {}
srcs = {"default": (main, "cnn_ft"), "scratch": (main, "cnn_real")}
for t in ("broad", "nonoise", "noambient"):
    srcs[t] = (L(f"abl_{t}"), "cnn_ft")
for name, (src, m) in srcs.items():
    abl[name] = {k: macro(src, m, k) for k in (4, 8)}
abl_tests = {}
for t in ("broad", "nonoise", "noambient"):
    for k in (4, 8):
        n = len(preds(srcs[t][0], "cnn_ft", k))
        if n >= 1:
            # restrict the default/scratch to the same seeds as the ablation (paired by seed)
            seeds = [s_ for s_, _ in preds(srcs[t][0], "cnn_ft", k)]
            def ge(src, m): return np.mean([np.nanmean(np.abs(p[:, SC_IDX] - Y[:, SC_IDX]), axis=1) for s_, p in preds(src, m, k) if s_ in seeds], axis=0)
            def gg(e): return np.array([np.nanmean(e[G == g]) for g in ug])
            for ref in ("default", "scratch"):
                d = gg(ge(srcs[t][0], "cnn_ft")) - gg(ge(*srcs[ref]))
                nz = d[d != 0]
                abl_tests[f"{t}_vs_{ref}|k{k}"] = dict(n_seeds=n, median_diff=float(np.median(d)), p=float(wilcoxon(nz).pvalue))
out["ablation"] = abl; out["ablation_tests"] = abl_tests
print({n: {k: (round(v[0], 3), round(v[1], 3), v[2]) for k, v in r.items()} for n, r in abl.items()})
for k_, v in abl_tests.items(): print(" ", k_, v)
json.dump(out, open(_R + "results/extra.json", "w"), indent=1, default=float)
