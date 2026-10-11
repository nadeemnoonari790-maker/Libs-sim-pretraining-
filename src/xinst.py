"""Cross-instrument pre-training test (dataset 2 -> dataset 1).

Question: for small-sample quantification on the deep-sea instrument, is pre-training on ~2,600 REAL spectra from a
different instrument (ChemLIBS, Mars atmosphere, silicate rocks) better or worse than pre-training on simulated spectra?

ChemLIBS spectra are re-binned onto the deep-sea pixel grid (321.3-559.3 nm, 1024 px); pixels falling in the ChemLIBS
spectrometer gaps (340.75-382.0, 469.0-473.05 nm) are set to zero. Labels in the deep-sea element order
(Cu, Zn, Fe, Pb, Ca, Mg, Mn, Al, Si, Ti, Ba): majors from oxides, Cu/Zn/Pb/Ba from the ppm columns (missing -> masked).
Pre-training (4000 steps, bs 128, lr 1e-3) and fine-tuning (500 steps, lr 3e-4) are identical to the simulation
pre-training arm; the deep-sea folds and label subsets are identical to the main experiment (same RNG seeds).
Controls: element-permuted ChemLIBS labels (same derangement as the simulation control).

  python xinst.py pretrain ; python xinst.py run
"""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, os, re, time, pickle, argparse
sys.path.insert(0, _SRC)
import numpy as np, pandas as pd, torch
from data import load_real, make_folds, metrics, TARGETS, CACHE, to_log
from models import Net, train, predict_samples, RealSampler, SCALE
from preprocess import preprocess
import data2

RES = _R + "results/"
C2 = data2.CACHE2
PERM = np.array([(i + 3) % 11 for i in range(11)])


def build():
    p = C2 + "chemlibs_on_deepsea.npz"
    if os.path.exists(p):
        return np.load(p)
    wl = np.load(CACHE + "real_blocks.npz")["wl"].astype(np.float64)
    e = np.concatenate([[wl[0] - (wl[1] - wl[0]) / 2], (wl[1:] + wl[:-1]) / 2, [wl[-1] + (wl[-1] - wl[-2]) / 2]])
    d = np.load(data2.ROOT + "data/chemlibs/chemlibs_mars5mj.npz")
    grid = d["grid"].astype(np.float64); S = d["spec"].astype(np.float64); files = [str(x) for x in d["names"]]
    fin = np.isfinite(S).all(0)
    seg_breaks = np.where(np.diff(fin.astype(int)) != 0)[0]
    starts = [0] + list(seg_breaks + 1)
    ends = list(seg_breaks + 1) + [len(grid)]
    X = np.zeros((len(S), len(wl)))
    for a, b in zip(starts, ends):
        if not fin[a]:
            continue
        g = grid[a:b]; Y = S[:, a:b]
        cum = np.concatenate([np.zeros((len(Y), 1)), np.cumsum((Y[:, 1:] + Y[:, :-1]) / 2 * np.diff(g)[None], 1)], 1)
        inside = (e[:-1] >= g[0]) & (e[1:] <= g[-1])
        idx = np.where(inside)[0]
        lo = np.stack([np.interp(e[idx], g, c) for c in cum]); hi = np.stack([np.interp(e[idx + 1], g, c) for c in cum])
        X[:, idx] = (hi - lo) / (e[idx + 1] - e[idx])[None]
    tgt = [re.search(r"_t\d+l(\d+)_(.+)_spect\.csv$", f, re.I).group(2).upper() for f in files]
    meta = pd.read_excel(data2.ROOT + "data/chemlibs/libs_metadata.xlsx").iloc[1:].reset_index(drop=True)
    meta["N"] = meta["PELLET NAME"].astype(str).str.upper().str.strip()
    O = meta[data2.OXIDES].apply(pd.to_numeric, errors="coerce"); tot = O.sum(axis=1, min_count=8)
    ok = (tot >= 85) & (tot <= 105) & O[data2.OXIDES[:9]].notna().all(axis=1)
    lab = {}
    for i, n in enumerate(meta["N"]):
        if not ok[i]:
            continue
        el = {e_: O.loc[i, ox] * data2.FACT[e_] for e_, ox in zip(data2.TARGETS2, data2.OXIDES)}
        for t in ("Cu", "Zn", "Pb", "Ba"):
            v = pd.to_numeric(meta.loc[i, t], errors="coerce")
            el[t] = v / 1e4 if (np.isfinite(v) and v < 2e5) else np.nan
        lab[n] = np.array([el.get(t, np.nan) for t in TARGETS], float)
    keep = np.array([t in lab for t in tgt])
    Y = preprocess(X[keep]); L = np.stack([lab[t] for t, k in zip(tgt, keep) if k])
    np.savez_compressed(p, Y=Y, L=L)
    print("built", Y.shape, flush=True)
    return np.load(p)


class LabSampler:
    """ChemLIBS real spectra on the deep-sea grid with masked labels."""
    def __init__(self, Y, L, perm=False):
        L = L[:, PERM] if perm else L
        self.Y = Y; self.y = np.where(np.isnan(L), 0, to_log(np.nan_to_num(L))).astype(np.float32); self.m = (~np.isnan(L)).astype(np.float32)

    def __call__(self, rng, bs):
        i = rng.integers(0, len(self.Y), bs)
        return self.Y[i] * SCALE, self.y[i], self.m[i]


def pretrain():
    z = build()
    for tag, perm in (("xreal", False), ("xrealperm", True)):
        for s in (1, 2, 3):
            p = C2 + f"pre_{tag}_s{s}.pt"
            if os.path.exists(p):
                continue
            t0 = time.time()
            net = train(Net(), LabSampler(z["Y"], z["L"], perm), 4000, 1e-3, 128, seed=s)
            torch.save(net.state_dict(), p); print(f"pretrained {tag} s{s} {time.time()-t0:.0f}s", flush=True)


def run(ks=(4, 8, 16, 32, 0), seeds=(0, 1, 2)):
    R = load_real(); NS = len(R["names"]); FOLD = make_folds(R["groups"]); Y = R["Ylog"]
    out = RES + "xinst_preds.pkl"
    res = pickle.load(open(out, "rb")) if os.path.exists(out) else {}
    for seed in seeds:
        for k in ks:
            for tag in ("xreal", "xrealperm"):
                key = f"cnn_{tag}|k{k}|s{seed}"
                if key in res:
                    continue
                P = np.full((NS, len(TARGETS)), np.nan); t0 = time.time()
                for f in range(5):
                    test_s = np.where(FOLD == f)[0]; train_all = np.where(FOLD != f)[0]
                    rng = np.random.default_rng(10000 * seed + 100 * f + k)
                    train_s = train_all if k == 0 else rng.choice(train_all, k, replace=False)
                    sd = 7919 * seed + f
                    net = Net(); net.load_state_dict(torch.load(C2 + f"pre_{tag}_s{seed + 1}.pt"))
                    net = train(net, RealSampler(R, train_s), 500, 3e-4, 64, seed=sd)
                    pr = predict_samples(net, R, test_s, seed=sd)
                    for s_ in test_s:
                        P[s_] = pr[s_]
                res[key] = P; pickle.dump(res, open(out, "wb"))
                print(f"{key} macro RMSE {metrics(P, Y)['macro_rmse']:.3f} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"build": build, "pretrain": pretrain, "run": run}[cmd]()
