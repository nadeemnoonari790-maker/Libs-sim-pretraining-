"""Aggregate results: macro RMSE per (method, k) over seeds, site-level bootstrap CIs, group-level paired Wilcoxon tests."""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, pickle, json, re
sys.path.insert(0, _SRC)
import numpy as np
from scipy.stats import wilcoxon
from data import *

R = load_real()
Y = R["Ylog"]; G = R["groups"]; UG = np.unique(G)
res = pickle.load(open(_R + "results/main_preds.pkl", "rb"))
keys = {}
for k_ in res:
    m, k, s = k_.split("|"); keys.setdefault((m, int(k[1:])), []).append((int(s[1:]), res[k_]))
METHODS = ["null", "pls", "cnn_real", "cnn_ft", "cnn_mix"]
KS = sorted({k for _, k in keys}, key=lambda x: (x == 0, x))      # 4, 8, 16, 32, all


def sq_err(pred):                       # (NS, len(SCORED)) squared error, NaN where label missing
    return (pred[:, SC_IDX] - Y[:, SC_IDX]) ** 2


def macro_rmse_from_sq(se, w=None):     # per-element RMSE over (weighted) samples, then macro mean
    w = np.ones(len(se)) if w is None else w
    out = []
    for j in range(se.shape[1]):
        v = ~np.isnan(se[:, j])
        out.append(np.sqrt(np.sum(w[v] * se[v, j]) / np.sum(w[v])))
    return float(np.mean(out))


def seed_mean_sq(m, k):
    return np.nanmean(np.stack([sq_err(p) for _, p in keys[(m, k)]]), axis=0)      # mean over seeds


rng = np.random.default_rng(0)
gid = {g: np.where(G == g)[0] for g in UG}
boot_idx = [rng.integers(0, len(UG), len(UG)) for _ in range(1000)]


def boot_ci(se):
    vals = []
    for bi in boot_idx:
        idx = np.concatenate([gid[UG[i]] for i in bi]); vals.append(macro_rmse_from_sq(se[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


table = {}
for (m, k), lst in sorted(keys.items(), key=lambda x: (METHODS.index(x[0][0]), x[0][1])):
    per_seed = [metrics(p, Y) for _, p in lst]
    se = seed_mean_sq(m, k)
    lo, hi = boot_ci(se)
    table[f"{m}|k{k}"] = dict(n_seeds=len(lst), macro_rmse=float(np.mean([x["macro_rmse"] for x in per_seed])),
                              sd_over_seeds=float(np.std([x["macro_rmse"] for x in per_seed])),
                              ci95=[lo, hi], macro_mae=float(np.mean([x["macro_mae"] for x in per_seed])),
                              macro_r2=float(np.mean([x["macro_r2"] for x in per_seed])),
                              per_element={e: float(np.mean([x[e] for x in per_seed])) for e in SCORED})


def grp_abs_err(m, k):
    ae = np.nanmean(np.stack([np.abs(p[:, SC_IDX] - Y[:, SC_IDX]) for _, p in keys[(m, k)]]), axis=0)   # (NS, n_el)
    per_sample = np.nanmean(ae, axis=1)
    return np.array([per_sample[gid[g]].mean() for g in UG])


tests = {}
for k in KS:
    for a, b in [("cnn_ft", "cnn_real"), ("cnn_ft", "pls"), ("cnn_ft", "cnn_mix"), ("cnn_mix", "cnn_real"), ("pls", "cnn_real")]:
        if (a, k) in keys and (b, k) in keys:
            ea, eb = grp_abs_err(a, k), grp_abs_err(b, k)
            try:
                p = float(wilcoxon(ea, eb).pvalue)
            except ValueError:
                p = float("nan")
            tests[f"{a}_vs_{b}|k{k}"] = dict(median_diff_abs_err=float(np.median(ea - eb)), frac_groups_a_better=float(np.mean(ea < eb)), p=p,
                                             n_groups=len(UG))
json.dump(dict(table=table, tests=tests), open(_R + "results/summary.json", "w"), indent=1)

print(f"{'method':9}" + "".join(f"{('k=all' if k == 0 else 'k=%d' % k):>22}" for k in KS))
for m in METHODS:
    row = f"{m:9}"
    for k in KS:
        t = table.get(f"{m}|k{k}")
        row += f"{(f'{t['macro_rmse']:.3f} [{t['ci95'][0]:.2f},{t['ci95'][1]:.2f}] n{t['n_seeds']}' if t else '-'):>22}"
    print(row)
print("\npaired group-level Wilcoxon (median diff in mean|err|; negative = first is better):")
for k_, v in tests.items():
    print(f"  {k_:28} d={v['median_diff_abs_err']:+.3f}  first-better in {v['frac_groups_a_better']*100:.0f}% of groups  p={v['p']:.3g}")
