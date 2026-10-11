"""Real-data loading, label handling, leave-site-out folds and metrics (protocol fixed before any model was evaluated).

Labels : log10(max(wt%, 0.01)); elements whose certified value is MISSING (alloy standards only certify Cu/Zn/Pb/Fe)
         are NaN and are masked out of both the training loss and every metric (never treated as zero).
Groups : samples from the same dive / site / alloy family are held out together (leave-site-out CV).
Scored : elements with >= 15 samples at >= 0.1 wt% spread over >= 10 groups (decided before training any model).
"""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import os, re, json, sys
import numpy as np

sys.path.insert(0, _SRC)
from preprocess import preprocess

CACHE = _R + "cache/"
TARGETS = ["Cu", "Zn", "Fe", "Pb", "Ca", "Mg", "Mn", "Al", "Si", "Ti", "Ba"]
SCORED = ["Cu", "Zn", "Fe", "Pb", "Ca", "Mg", "Mn", "Al", "Si", "Ba"]
SC_IDX = [TARGETS.index(e) for e in SCORED]
FLOOR = 0.01
NPIX = 1024
# Robustness switch (default OFF): mask detector-edge bands in BOTH real and synthetic spectra, then re-normalise.
CROP = os.environ.get("LIBS_CROP", "0") == "1"
CROP_SUFFIX = "_crop" if CROP else ""
CROP_BANDS_NM = (333.0, 545.0)


def apply_crop(Y):
    wl = np.load(CACHE + "real_blocks.npz")["wl"]
    m = ((wl >= CROP_BANDS_NM[0]) & (wl <= CROP_BANDS_NM[1])).astype(np.float32)
    Y = np.asarray(Y, np.float32) * m[None, :]
    n = np.linalg.norm(Y, axis=1, keepdims=True)
    return (Y / np.maximum(n, 1e-12)).astype(np.float32)


def group_key(n):
    m = re.match(r"^(HPD\d{3,4}|2K\d{4}|6K\d{3})", n)
    return m.group(1) if m else n


def to_log(c):
    return np.log10(np.maximum(c, FLOOR))


def load_real():
    D = np.load(CACHE + "real_blocks.npz")
    p = CACHE + "real_pre.npy"
    if os.path.exists(p):
        Y = np.load(p)
    else:
        Y = preprocess(D["X"]); np.save(p, Y)
    names = [str(n) for n in D["names"]]
    comp = json.load(open(CACHE + "real_blocks_composition.json"))
    C = np.array([[comp[n].get(e, np.nan) for e in TARGETS] for n in names], float)
    sid = D["sample_idx"].astype(int)
    order = np.argsort(sid, kind="stable")
    Y, sid = Y[order], sid[order]
    if CROP:
        Y = apply_crop(Y)
    start = np.searchsorted(sid, np.arange(len(names)))
    count = np.bincount(sid, minlength=len(names))
    groups = np.array([group_key(n) for n in names])
    Ylog = np.where(np.isnan(C), np.nan, to_log(np.nan_to_num(C)))
    return dict(Y=Y, sid=sid, start=start, count=count, names=names, C=C, Ylog=Ylog, groups=groups)


def make_folds(groups, nfold=5, seed=0):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    size = np.bincount(inv)
    order = np.argsort(-size + rng.random(len(size)) * 0.5)   # large groups first, random tie-break
    fold_of_group = np.zeros(len(ug), int); load = np.zeros(nfold)
    for g in order:
        f = int(np.argmin(load)); fold_of_group[g] = f; load[f] += size[g]
    return fold_of_group[inv]


def metrics(pred, y):
    """pred, y: (n_samples, NT) in log10 units; y may contain NaN. Returns per-scored-element RMSE/MAE and macro averages."""
    out = {}
    rm, ma, r2 = [], [], []
    for e, j in zip(SCORED, SC_IDX):
        v = ~np.isnan(y[:, j])
        d = pred[v, j] - y[v, j]
        rm.append(float(np.sqrt(np.mean(d ** 2)))); ma.append(float(np.mean(np.abs(d))))
        r2.append(float(1 - np.sum(d ** 2) / np.sum((y[v, j] - y[v, j].mean()) ** 2)))
        out[e] = rm[-1]
    out["macro_rmse"] = float(np.mean(rm)); out["macro_mae"] = float(np.mean(ma)); out["macro_r2"] = float(np.mean(r2))
    return out


def per_sample_err(pred, y):
    """Mean absolute log10 error over scored elements, per sample (NaN labels ignored)."""
    d = np.abs(pred[:, SC_IDX] - y[:, SC_IDX])
    return np.nanmean(d, axis=1)
