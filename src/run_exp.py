"""Main experiment: leave-site-out 5-fold CV, label-budget sweep, five methods. Resumable (pickle of sample-level predictions).

Methods
  null     predict the training-fold mean
  pls      per-element PLS1 on real blocks (n_components chosen by inner site-grouped CV)
  cnn_real CNN trained from scratch on k real training samples
  cnn_ft   synthetic-pretrained CNN, fine-tuned on the same k real samples
  cnn_mix  CNN trained from scratch on 50% real / 50% synthetic batches
k = number of real training SAMPLES (0 => all samples of the training folds).
"""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, os, time, pickle, copy, argparse
sys.path.insert(0, _SRC)
import numpy as np, torch
from sklearn.cross_decomposition import PLSRegression
from data import *
from models import *

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--ks", type=int, nargs="+", default=[0, 32, 16, 8, 4])
ap.add_argument("--methods", nargs="+", default=["null", "pls", "cnn_real", "cnn_ft", "cnn_mix"])
ap.add_argument("--pre_tag", default="informed")
ap.add_argument("--out", default=_R + "results/main_preds.pkl")
ap.add_argument("--steps_real", type=int, default=1500); ap.add_argument("--steps_ft", type=int, default=500)
a = ap.parse_args()

R = load_real()
NS = len(R["names"])
FOLD = make_folds(R["groups"])
Y = R["Ylog"]
res = pickle.load(open(a.out, "rb")) if os.path.exists(a.out) else {}
SYN = {}


def log(s):
    print(s, flush=True)


def synth(seed):
    if seed not in SYN:
        z = np.load(CACHE + f"synth_{a.pre_tag}_s{seed + 1}.npz")
        SYN[seed] = SynthSampler(z["Y"], z["L"])
    return SYN[seed]


def pretrained(seed):
    net = Net(); net.load_state_dict(torch.load(CACHE + f"pre_{a.pre_tag}_s{seed + 1}{CROP_SUFFIX}.pt")); return net


def block_idx(samples, per, rng):
    out = []
    for s in samples:
        n = R["count"][s]; out.append(R["start"][s] + rng.choice(n, min(per, n), replace=False))
    return np.concatenate(out), np.concatenate([[s] * min(per, R["count"][s]) for s in samples])


def run_pls(train_s, test_s, seed):
    rng = np.random.default_rng(1000 + seed)
    bi, bs_ = block_idx(train_s, 60, rng)
    X = R["Y"][bi] * SCALE
    grp = R["groups"][bs_]
    pred = np.full((NS, len(TARGETS)), np.nan)
    ncs = [4, 8, 12, 16, 24]
    tb = np.concatenate([R["start"][s] + rng.choice(R["count"][s], min(120, R["count"][s]), replace=False) for s in test_s])
    tid = np.concatenate([[s] * min(120, R["count"][s]) for s in test_s])
    for j in range(len(TARGETS)):
        yb = Y[bs_, j]; v = ~np.isnan(yb)
        if v.sum() < 50 or len(set(grp[v])) < 3:
            pred[test_s, j] = np.nanmean(yb) if v.any() else 0.0
            continue
        Xv, yv, gv = X[v], yb[v], grp[v]
        ug = np.unique(gv); fo = {g: i % 3 for i, g in enumerate(rng.permutation(ug))}
        ff = np.array([fo[g] for g in gv]); err = np.zeros(len(ncs))
        for f in range(3):
            m = PLSRegression(n_components=max(ncs), scale=False).fit(Xv[ff != f], yv[ff != f])
            for q, nc in enumerate(ncs):
                B = m.x_rotations_[:, :nc] @ m.y_loadings_[0, :nc]
                p = (Xv[ff == f] - m._x_mean) @ B + float(np.ravel(m._y_mean)[0])
                err[q] += np.mean((p - yv[ff == f]) ** 2)
        nc = ncs[int(np.argmin(err))]
        m = PLSRegression(n_components=nc, scale=False).fit(Xv, yv)
        pb = m.predict(R["Y"][tb] * SCALE).ravel()
        for s in test_s:
            pred[s, j] = pb[tid == s].mean()
    return pred


def one(method, k, seed):
    pred = np.full((NS, len(TARGETS)), np.nan)
    for f in range(5):
        test_s = np.where(FOLD == f)[0]; train_all = np.where(FOLD != f)[0]
        rng = np.random.default_rng(10000 * seed + 100 * f + k)
        train_s = train_all if k == 0 else rng.choice(train_all, k, replace=False)
        t0 = time.time(); sd = 7919 * seed + f
        if method == "null":
            mu = np.nanmean(Y[train_s], axis=0); pred[test_s] = mu
            continue
        if method == "pls":
            p = run_pls(train_s, test_s, seed); pred[test_s] = p[test_s]
        else:
            rs = RealSampler(R, train_s)
            if method == "cnn_real":
                net = train(Net(), rs, a.steps_real, 1e-3, 64, seed=sd)
            elif method == "cnn_ft":
                net = train(pretrained(seed), rs, a.steps_ft, 3e-4, 64, seed=sd)
            elif method == "cnn_head":      # frozen conv feature extractor, train the head only
                net = train(pretrained(seed), rs, a.steps_ft, 1e-3, 64, seed=sd, freeze_f=True)
            elif method == "cnn_mix":
                net = train(Net(), MixSampler(rs, synth(seed), 0.5), a.steps_real, 1e-3, 64, seed=sd)
            pr = predict_samples(net, R, test_s, seed=sd)
            for s in test_s:
                pred[s] = pr[s]
        log(f"  {method} k={k} seed={seed} fold={f} ({len(train_s)} train samples) {time.time()-t0:.0f}s")
    return pred


t00 = time.time()
for seed in a.seeds:
    for k in a.ks:
        for method in a.methods:
            if method == "null" and seed > 0:
                continue
            key = f"{method}|k{k}|s{seed}"
            if key in res:
                continue
            res[key] = one(method, k, seed)
            pickle.dump(res, open(a.out, "wb"))
            m = metrics(res[key], Y)
            log(f"[{time.time()-t00:.0f}s] {key}: macro RMSE {m['macro_rmse']:.3f}  MAE {m['macro_mae']:.3f}  R2 {m['macro_r2']:.3f}")
