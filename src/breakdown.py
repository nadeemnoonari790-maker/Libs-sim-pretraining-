"""Error by sample type, and within-type rank correlation (does the model quantify, or just recognise the sample class?)."""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, pickle, json
sys.path.insert(0, _SRC)
import numpy as np
from scipy.stats import spearmanr
from data import *

R = load_real(); Y = R["Ylog"]; names = R["names"]
TYPE = np.array(["alloy" if n.startswith(("C", "31X")) else ("ref" if n.startswith("J") else "natural") for n in names])
res = pickle.load(open(_R + "results/main_preds.pkl", "rb"))


def ser(m, k):
    return [v for key, v in res.items() if key.startswith(f"{m}|k{k}|")]


def mean_pred(m, k):
    s = ser(m, k)
    return np.mean(s, axis=0) if s else None


def rmse_type(P, t):
    idx = TYPE == t; out = []
    for j in SC_IDX:
        v = idx & ~np.isnan(Y[:, j])
        if v.sum() >= 3:
            out.append(np.sqrt(np.mean((P[v, j] - Y[v, j]) ** 2)))
    return float(np.mean(out)) if out else np.nan


print("types:", {t: int((TYPE == t).sum()) for t in ("alloy", "ref", "natural")})
out = {}
print(f"\n{'':9}{'':>6}" + "".join(f"{t:>10}" for t in ("alloy", "ref", "natural")) + "   (macro RMSE, log10 wt%, per type; seed-mean predictions)")
for k in (4, 8, 0):
    for m in ("null", "cnn_real", "cnn_ft"):
        P = mean_pred(m, k)
        if P is None:
            continue
        row = {t: rmse_type(P, t) for t in ("alloy", "ref", "natural")}; out[f"{m}|k{k}"] = row
        print(f"{m:9}k={k:<4}" + "".join(f"{row[t]:>10.3f}" for t in row))

print("\nWithin-type Spearman rho (predicted vs certified) on NATURAL samples, per element [n valid]:")
els = ["Cu", "Zn", "Fe", "Pb", "Si", "Al", "Ca"]
print(f"{'':16}" + "".join(f"{e:>12}" for e in els))
nat = TYPE == "natural"; wr = {}
for k in (8, 0):
    for m in ("cnn_real", "cnn_ft"):
        P = mean_pred(m, k)
        if P is None:
            continue
        row = []
        for e in els:
            j = TARGETS.index(e); v = nat & ~np.isnan(Y[:, j])
            r = spearmanr(P[v, j], Y[v, j]).statistic if v.sum() > 5 and np.std(Y[v, j]) > 0 else np.nan
            row.append(r)
        wr[f"{m}|k{k}"] = dict(zip(els, map(float, row)))
        print(f"{m+' k='+str(k):16}" + "".join(f"{r:>12.2f}" for r in row))
json.dump(dict(rmse_by_type=out, within_natural_spearman=wr, n_natural=int(nat.sum())), open(_R + "results/breakdown.json", "w"), indent=1)
