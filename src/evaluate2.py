"""Dataset 2 aggregation: macro RMSE per (method, k) over seeds, suite-level bootstrap CIs, paired suite-level Wilcoxon
tests (same statistics as dataset 1), relative gains, per-element breakdown; plus the cross-instrument test on dataset 1."""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, pickle, json, os
sys.path.insert(0, _SRC)
import numpy as np
from scipy.stats import wilcoxon
from data2 import load_real2, TARGETS2
from exp2 import metrics2

RES = _R + "results/"
R = load_real2(); Y = R["Ylog"]; G = R["groups"]; UG = np.unique(G)
gid = {g: np.where(G == g)[0] for g in UG}
res = pickle.load(open(RES + "main2_preds.pkl", "rb"))
keys = {}
for k_, p in res.items():
    m, k, s = k_.split("|"); keys.setdefault((m, int(k[1:])), []).append((int(s[1:]), p))
METHODS = ["null", "pls", "cnn_real", "cnn_ft", "cnn_elperm", "cnn_head", "cnn_mix"]
KS = sorted({k for _, k in keys}, key=lambda x: (x == 0, x))


def sq(p): return (p - Y) ** 2


def macro_from_sq(se):
    return float(np.mean([np.sqrt(np.nanmean(se[:, j])) for j in range(se.shape[1])]))


rng = np.random.default_rng(0)
boot = [rng.integers(0, len(UG), len(UG)) for _ in range(1000)]


def ci(se):
    v = [macro_from_sq(se[np.concatenate([gid[UG[i]] for i in b])]) for b in boot]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


table = {}
for (m, k), lst in keys.items():
    ps = [metrics2(p, Y) for _, p in lst]
    se = np.nanmean(np.stack([sq(p) for _, p in lst]), 0)
    table[f"{m}|k{k}"] = dict(n_seeds=len(lst), macro_rmse=float(np.mean([x["macro_rmse"] for x in ps])),
                              sd_over_seeds=float(np.std([x["macro_rmse"] for x in ps])), ci95=ci(se),
                              macro_mae=float(np.mean([x["macro_mae"] for x in ps])), macro_r2=float(np.mean([x["macro_r2"] for x in ps])),
                              per_element={e: float(np.mean([x[e] for x in ps])) for e in TARGETS2})


def gerr(m, k):
    ae = np.nanmean(np.stack([np.abs(p - Y) for _, p in keys[(m, k)]]), 0)
    ps = np.nanmean(ae, 1)
    return np.array([ps[gid[g]].mean() for g in UG])


tests = {}
for k in KS:
    for a, b in [("cnn_ft", "cnn_real"), ("cnn_ft", "pls"), ("cnn_ft", "cnn_elperm"), ("cnn_elperm", "cnn_real"),
                 ("cnn_mix", "cnn_real"), ("cnn_ft", "cnn_mix"), ("cnn_head", "cnn_real"), ("cnn_ft", "cnn_head"), ("pls", "cnn_real")]:
        if (a, k) in keys and (b, k) in keys:
            d = gerr(a, k) - gerr(b, k)
            tests[f"{a}_vs_{b}|k{k}"] = dict(median_diff=float(np.median(d)), frac_first_better=float(np.mean(d < 0)),
                                             p=float(wilcoxon(d[d != 0]).pvalue), n_groups=int(len(UG)))
gain = {}
for k in KS:
    if ("cnn_ft", k) in keys and ("cnn_real", k) in keys:
        a, b = table[f"cnn_ft|k{k}"]["macro_rmse"], table[f"cnn_real|k{k}"]["macro_rmse"]
        gain[k] = dict(scratch=b, ft=a, rel=(b - a) / b)
out = dict(table=table, tests=tests, gain=gain, n_targets=len(R["names"]), n_groups=len(UG), n_spectra=int(len(R["Y"])))

# ---------------- zero-shot
if os.path.exists(RES + "zeroshot2.json"):
    out["zeroshot"] = json.load(open(RES + "zeroshot2.json"))

# ---------------- cross-instrument test on dataset 1
if os.path.exists(RES + "xinst_preds.pkl"):
    from data import load_real, metrics, SC_IDX
    R1 = load_real(); Y1 = R1["Ylog"]; G1 = R1["groups"]; U1 = np.unique(G1)
    xi = pickle.load(open(RES + "xinst_preds.pkl", "rb")); mp = pickle.load(open(RES + "main_preds.pkl", "rb"))
    allp = {**mp, **xi}
    x = {}

    def g1(m, k):
        ks = [kk for kk in allp if kk.startswith(f"{m}|k{k}|")]
        ae = np.nanmean(np.stack([np.abs(allp[kk][:, SC_IDX] - Y1[:, SC_IDX]) for kk in ks]), 0)
        ps = np.nanmean(ae, 1)
        return np.array([ps[G1 == g].mean() for g in U1])
    for k in (4, 8, 16, 32, 0):
        row = {}
        for m in ("cnn_real", "cnn_ft", "cnn_xreal", "cnn_xrealperm"):
            ks = [kk for kk in allp if kk.startswith(f"{m}|k{k}|")]
            if ks:
                v = [metrics(allp[kk], Y1)["macro_rmse"] for kk in ks]
                row[m] = dict(mean=float(np.mean(v)), sd=float(np.std(v)), n=len(v))
        for a, b in (("cnn_xreal", "cnn_real"), ("cnn_ft", "cnn_xreal"), ("cnn_xreal", "cnn_xrealperm")):
            if a in row and b in row:
                d = g1(a, k) - g1(b, k)
                row[f"{a}_vs_{b}"] = dict(median_diff=float(np.median(d)), frac_first_better=float(np.mean(d < 0)), p=float(wilcoxon(d[d != 0]).pvalue))
        x[k] = row
    out["xinst"] = x

# ---------------- attribution
if os.path.exists(RES + "sal2.json"):
    out["sal2"] = json.load(open(RES + "sal2.json"))
json.dump(out, open(RES + "summary2.json", "w"), indent=1)

print(f"{'method':11}" + "".join(f"{('all' if k == 0 else k):>24}" for k in KS))
for m in METHODS:
    row = f"{m:11}"
    for k in KS:
        t = table.get(f"{m}|k{k}")
        cell = "%.3f±%.3f [%.2f,%.2f]" % (t["macro_rmse"], t["sd_over_seeds"], t["ci95"][0], t["ci95"][1]) if t else "-"
        row += f"{cell:>24}"
    print(row)
for k, v in gain.items():
    print(f"gain k={k}: {v['scratch']:.3f} -> {v['ft']:.3f} ({100 * v['rel']:.1f}%)")
for k_, v in tests.items():
    print(f"  {k_:28} d={v['median_diff']:+.3f} first-better {100 * v['frac_first_better']:.0f}% p={v['p']:.2g}")
if "xinst" in out:
    for k, r in out["xinst"].items():
        print("xinst k", k, {m: (round(v["mean"], 3) if "mean" in v else (round(v["median_diff"], 3), round(v["p"], 4))) for m, v in r.items()})
