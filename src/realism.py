"""Label-free realism diagnostics: classifier two-sample test (C2ST) real vs synthetic, plus simple spectral statistics."""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, json, os
sys.path.insert(0, _SRC)
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from data import *

R = load_real()
rng = np.random.default_rng(0)
NS = len(R["names"])
# real: equal weight per sample (up to 60 blocks each), then cap at 4000
ri = np.concatenate([R["start"][s] + rng.choice(R["count"][s], min(60, R["count"][s]), replace=False) for s in range(NS)])
ri = rng.choice(ri, min(4000, len(ri)), replace=False)
Xr = R["Y"][ri]
tags = sys.argv[1:] or ["informed_s1", "broad_s1", "nonoise_s1", "noambient_s1"]
out = {}


def feats(X):
    return X[:, ::4]          # 256 channels


def stats(X):
    mx = X.max(1, keepdims=True)
    return dict(frac_pix_gt5pct=float((X > 0.05 * mx).mean()), frac_pix_gt20pct=float((X > 0.2 * mx).mean()),
                peak_pos_nm=float(np.median(np.load(CACHE + "real_blocks.npz")["wl"][X.argmax(1)])))


out["real" + CROP_SUFFIX] = stats(Xr)
for tag in tags:
    z = np.load(CACHE + f"synth_{tag}.npz")
    Xs = z["Y"][rng.choice(len(z["Y"]), len(Xr), replace=False)]
    if CROP:
        Xs = apply_crop(Xs)
    X = np.concatenate([feats(Xr), feats(Xs)]); y = np.r_[np.zeros(len(Xr)), np.ones(len(Xs))]
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, random_state=0)
    acc = cross_val_score(clf, X, y, cv=StratifiedKFold(4, shuffle=True, random_state=0), scoring="accuracy")
    mr = Xr.mean(0); ms = Xs.mean(0)
    out[tag + CROP_SUFFIX] = dict(c2st_acc=float(acc.mean()), c2st_sd=float(acc.std()), mean_spec_corr=float(np.corrcoef(mr, ms)[0, 1]), **stats(Xs))
    print(tag, out[tag + CROP_SUFFIX], flush=True)
os.makedirs(_R + "results", exist_ok=True)
fn = _R + "results/realism.json"
old = json.load(open(fn)) if os.path.exists(fn) else {}
old.update(out); json.dump(old, open(fn, "w"), indent=1)
