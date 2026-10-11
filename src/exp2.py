"""Dataset 2 (ChemLIBS) pipeline: pre-training and the leave-suite-out label-budget experiment.
Identical network, optimiser, step counts, augmentation, preprocessing and fold logic as dataset 1; nothing tuned on dataset 2.

  python exp2.py pretrain --tag synth2 --seed 1        (also: --tag elperm2 = element-permuted labels control)
  python exp2.py run --seeds 0 1 2 --ks 4 8 16 32 128 0
Methods: null, pls, cnn_real (scratch), cnn_ft (simulation-pretrained, fine-tuned), cnn_elperm (fine-tuned from the
element-permuted-label pretraining), cnn_mix (50/50 real/synthetic batches from scratch).
"""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, os, time, json, pickle, argparse
sys.path.insert(0, _SRC)
import numpy as np, torch
from sklearn.cross_decomposition import PLSRegression
from data import make_folds
from data2 import load_real2, TARGETS2, CACHE2, to_log
import models
from models import Net, train, predict_samples, SCALE, MixSampler

NT2 = len(TARGETS2)
RES = _R + "results/"
PERM2 = np.array([(i + 3) % NT2 for i in range(NT2)])      # derangement: no element keeps its own label
assert not np.any(PERM2 == np.arange(NT2))


class RealSampler2:
    def __init__(self, R, samples):
        self.R, self.samples = R, np.asarray(samples)

    def __call__(self, rng, bs):
        s = rng.choice(self.samples, bs)
        b = self.R["start"][s] + (rng.random(bs) * self.R["count"][s]).astype(int)
        y = self.R["Ylog"][s]
        return self.R["Y"][b] * SCALE, np.nan_to_num(y), (~np.isnan(y)).astype(np.float32)


class SynthSampler2:
    def __init__(self, Y, L):
        self.Y, self.y = Y, to_log(L).astype(np.float32)

    def __call__(self, rng, bs):
        i = rng.integers(0, len(self.Y), bs)
        return self.Y[i] * SCALE, self.y[i], np.ones((bs, NT2), np.float32)


def metrics2(pred, y, els=TARGETS2):
    rm, ma, r2, out = [], [], [], {}
    for e in els:
        j = TARGETS2.index(e); v = ~np.isnan(y[:, j]) & ~np.isnan(pred[:, j])
        d = pred[v, j] - y[v, j]
        rm.append(float(np.sqrt(np.mean(d ** 2)))); ma.append(float(np.mean(np.abs(d))))
        r2.append(float(1 - np.sum(d ** 2) / np.sum((y[v, j] - y[v, j].mean()) ** 2))); out[e] = rm[-1]
    out.update(macro_rmse=float(np.mean(rm)), macro_mae=float(np.mean(ma)), macro_r2=float(np.mean(r2)))
    return out


def load_synth(tag, seed):
    if tag == "elperm2":
        z = np.load(CACHE2 + f"synth2_s{seed}.npz"); return z["Y"], z["L"][:, PERM2]
    z = np.load(CACHE2 + f"{tag}_s{seed}.npz"); return z["Y"], z["L"]


def pretrain(tag, seed, steps=4000, bs=128, lr=1e-3):
    out = CACHE2 + f"pre_{tag}_s{seed}.pt"
    if os.path.exists(out):
        print("exists", out); return
    R = load_real2()
    Ys, L = load_synth(tag, seed)
    t0 = time.time()
    net = train(Net(NT2), SynthSampler2(Ys, L), steps, lr, bs, seed=seed, log=lambda s: print(s, flush=True))
    torch.save(net.state_dict(), out)
    pr = predict_samples(net, R, range(len(R["names"])), seed=0)
    P = np.stack([pr[s] for s in range(len(R["names"]))])
    m = metrics2(P, R["Ylog"])
    fn = RES + "zeroshot2.json"
    d = json.load(open(fn)) if os.path.exists(fn) else {}
    d[f"{tag}_s{seed}"] = dict(metrics=m, train_s=time.time() - t0)
    json.dump(d, open(fn, "w"), indent=1)
    np.save(CACHE2 + f"zs_pred_{tag}_s{seed}.npy", P)
    print(f"[{tag} s{seed}] zero-shot macro RMSE {m['macro_rmse']:.3f} R2 {m['macro_r2']:.3f} ({time.time()-t0:.0f}s) "
          + " ".join(f"{e}={m[e]:.2f}" for e in TARGETS2), flush=True)


def run(seeds, ks, methods, out, steps_real=1500, steps_ft=500):
    R = load_real2(); NS = len(R["names"]); Y = R["Ylog"]
    FOLD = make_folds(R["groups"])
    res = pickle.load(open(out, "rb")) if os.path.exists(out) else {}
    SYN = {}

    def pre(tag, seed):
        net = Net(NT2); net.load_state_dict(torch.load(CACHE2 + f"pre_{tag}_s{seed + 1}.pt")); return net

    def synth(seed):
        if seed not in SYN:
            SYN[seed] = SynthSampler2(*load_synth("synth2", seed + 1))
        return SYN[seed]

    def run_pls(train_s, test_s, seed):
        rng = np.random.default_rng(1000 + seed)
        bi = np.concatenate([R["start"][s] + np.arange(R["count"][s]) for s in train_s])
        bs_ = np.concatenate([[s] * R["count"][s] for s in train_s])
        X = R["Y"][bi] * SCALE; grp = R["groups"][bs_]
        pred = np.full((NS, NT2), np.nan)
        tb = np.concatenate([R["start"][s] + np.arange(R["count"][s]) for s in test_s])
        tid = np.concatenate([[s] * R["count"][s] for s in test_s])
        for j in range(NT2):
            yb = Y[bs_, j]; v = ~np.isnan(yb)
            if v.sum() < 6:
                pred[test_s, j] = np.nanmean(yb) if v.any() else 0.0
                continue
            Xv, yv, gv = X[v], yb[v], grp[v]
            # inner 3-fold CV grouped by suite; with < 3 suites fall back to grouping by target (both spectra together)
            unit = gv if len(set(gv)) >= 3 else np.array([str(x) for x in bs_[v]])
            ug = np.unique(unit); fo = {g: i % 3 for i, g in enumerate(rng.permutation(ug))}
            ff = np.array([fo[g] for g in unit])
            n_min = min(int((ff != f).sum()) for f in range(3))
            ncs = [nc for nc in [1, 2, 4, 8, 12, 16, 24] if nc < n_min] or [1]
            err = np.zeros(len(ncs))
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
        pred = np.full((NS, NT2), np.nan)
        for f in range(5):
            test_s = np.where(FOLD == f)[0]; train_all = np.where(FOLD != f)[0]
            rng = np.random.default_rng(10000 * seed + 100 * f + k)
            train_s = train_all if k == 0 else rng.choice(train_all, k, replace=False)
            t0 = time.time(); sd = 7919 * seed + f
            if method == "null":
                pred[test_s] = np.nanmean(Y[train_s], axis=0); continue
            if method == "pls":
                pred[test_s] = run_pls(train_s, test_s, seed)[test_s]
            else:
                rs = RealSampler2(R, train_s)
                if method == "cnn_real":
                    net = train(Net(NT2), rs, steps_real, 1e-3, 64, seed=sd)
                elif method == "cnn_ft":
                    net = train(pre("synth2", seed), rs, steps_ft, 3e-4, 64, seed=sd)
                elif method == "cnn_elperm":
                    net = train(pre("elperm2", seed), rs, steps_ft, 3e-4, 64, seed=sd)
                elif method == "cnn_head":
                    net = train(pre("synth2", seed), rs, steps_ft, 1e-3, 64, seed=sd, freeze_f=True)
                elif method == "cnn_mix":
                    net = train(Net(NT2), MixSampler(rs, synth(seed), 0.5), steps_real, 1e-3, 64, seed=sd)
                pr = predict_samples(net, R, test_s, seed=sd)
                for s in test_s:
                    pred[s] = pr[s]
            print(f"  {method} k={k} seed={seed} fold={f} ({len(train_s)} train) {time.time()-t0:.0f}s", flush=True)
        return pred

    t00 = time.time()
    for seed in seeds:
        for k in ks:
            for method in methods:
                if method == "null" and seed > 0:
                    continue
                key = f"{method}|k{k}|s{seed}"
                if key in res:
                    continue
                res[key] = one(method, k, seed)
                pickle.dump(res, open(out, "wb"))
                m = metrics2(res[key], Y)
                print(f"[{time.time()-t00:.0f}s] {key}: macro RMSE {m['macro_rmse']:.3f} R2 {m['macro_r2']:.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--tag", default="synth2"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32, 128, 0])
    ap.add_argument("--methods", nargs="+", default=["null", "pls", "cnn_real", "cnn_ft", "cnn_elperm", "cnn_mix"])
    ap.add_argument("--out", default=RES + "main2_preds.pkl")
    a = ap.parse_args()
    if a.cmd == "pretrain":
        pretrain(a.tag, a.seed)
    else:
        run(a.seeds, a.ks, a.methods, a.out)
