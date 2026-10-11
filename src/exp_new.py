"""Three added analyses (all resumable).
 sal  : input-gradient attribution -> do fine-tuned nets use the TARGET element's own emission lines? (scratch / pretrained / element-permuted-pretrained)
 xmat : cross-matrix transfer. Train on lab alloys(+reference pellets), test on natural samples. (A) alloy+ref, (B) alloy only.
 mlp  : second architecture (MLP) -- pretrain vs scratch under the same leave-site-out protocol.
"""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, os, time, pickle, argparse, sqlite3
sys.path.insert(0, _SRC)
import numpy as np, torch, torch.nn as nn
from data import *
from models import *

ap = argparse.ArgumentParser()
ap.add_argument("--part", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
a = ap.parse_args()
RES = _R + "results/"
R = load_real(); NS = len(R["names"]); FOLD = make_folds(R["groups"]); Y = R["Ylog"]
TYPE = np.array(["alloy" if n.startswith(("C", "31X")) else ("ref" if n.startswith("J") else "natural") for n in R["names"]])
wl = np.load(CACHE + "real_blocks.npz")["wl"]
log = lambda s: print(s, flush=True)


def load(path): return pickle.load(open(path, "rb")) if os.path.exists(path) else {}
def pre(tag, seed):
    net = Net(); net.load_state_dict(torch.load(CACHE + f"pre_{tag}_s{seed + 1}.pt")); return net


# ------------------------------------------------------------------ line windows
def line_windows():
    con = sqlite3.connect("file:" + _R + "data/atomic_lines.db?mode=ro", uri=True)
    W = {}
    for e in TARGETS:
        rows = con.execute("select wavelength_nm,aki,gk,ek_ev from lines where element=? and sp_num in (1,2) and wavelength_nm between ? and ? "
                           "and aki>0 and gk>0 and ek_ev<12", (e, wl[0], wl[-1])).fetchall()
        r = np.array(rows, float)
        w = r[:, 1] * r[:, 2] * np.exp(-r[:, 3] / 1.0)       # Boltzmann-weighted strength at kT = 1 eV
        top = r[np.argsort(-w)[:12], 0]
        m = np.zeros(NPIX, bool)
        for l in top:
            m |= np.abs(wl - l) <= 0.35                         # +- 0.35 nm (~1.5 pixel)
        W[e] = m
    return W


@torch.no_grad()
def _noop(): pass


def saliency(net, test_s, seed, per=30):
    """Mean |d output_e / d x * x| over test blocks, per target element -> (NT, NPIX)."""
    rng = np.random.default_rng(seed); net.eval()
    idx = np.concatenate([R["start"][s] + rng.choice(R["count"][s], min(per, R["count"][s]), replace=False) for s in test_s])
    x = torch.from_numpy((R["Y"][idx] * SCALE).astype(np.float32))[:, None, :].requires_grad_(True)
    out = net(x); S = np.zeros((NT, NPIX))
    for j in range(NT):
        g, = torch.autograd.grad(out[:, j].sum(), x, retain_graph=True)
        S[j] = (g * x).abs().squeeze(1).mean(0).detach().numpy()
    return S


def part_sal():
    path = RES + "sal.pkl"; res = load(path); W = line_windows()
    for seed in a.seeds:
        for f in range(5):
            key = f"s{seed}|f{f}"
            if key in res: continue
            test_s = np.where(FOLD == f)[0]; train_s = np.where(FOLD != f)[0]; rs = RealSampler(R, train_s); sd = 7919 * seed + f
            t0 = time.time(); out = {}
            out["scratch"] = saliency(train(Net(), rs, 1500, 1e-3, 64, seed=sd), test_s, sd)
            out["pretrained"] = saliency(train(pre("informed", seed), rs, 500, 3e-4, 64, seed=sd), test_s, sd)
            out["elperm"] = saliency(train(pre("elperm", seed), rs, 500, 3e-4, 64, seed=sd), test_s, sd)
            out["pre_only"] = saliency(pre("informed", seed), test_s, sd)           # synthetic pre-training, no fine-tuning
            res[key] = out; pickle.dump(res, open(path, "wb")); log(f"sal {key} {time.time()-t0:.0f}s")


# ------------------------------------------------------------------ cross-matrix
def part_xmat():
    path = RES + "xmat.pkl"; res = load(path)
    test_s = np.where(TYPE == "natural")[0]
    for seed in a.seeds:
        for setting, train_types in (("A_alloy+ref", ("alloy", "ref")), ("B_alloy", ("alloy",))):
            train_s = np.where(np.isin(TYPE, train_types))[0]; rs = RealSampler(R, train_s)
            for method in ("scratch", "ft", "elperm"):
                key = f"{setting}|{method}|s{seed}"
                if key in res: continue
                sd = 7919 * seed + 11; t0 = time.time()
                if method == "scratch": net = train(Net(), rs, 1500, 1e-3, 64, seed=sd)
                elif method == "ft": net = train(pre("informed", seed), rs, 500, 3e-4, 64, seed=sd)
                else: net = train(pre("elperm", seed), rs, 500, 3e-4, 64, seed=sd)
                pr = predict_samples(net, R, test_s, seed=sd); P = np.full((NS, NT), np.nan)
                for s in test_s: P[s] = pr[s]
                res[key] = P; pickle.dump(res, open(path, "wb")); log(f"xmat {key} {time.time()-t0:.0f}s")


# ------------------------------------------------------------------ MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NPIX, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, NT))
    def forward(self, x): return self.net(x.squeeze(1))


def part_mlp():
    path = RES + "mlp_preds.pkl"; res = load(path)
    for seed in a.seeds:
        pp = CACHE + f"pre_mlp_s{seed + 1}.pt"
        if not os.path.exists(pp):
            z = np.load(CACHE + f"synth_informed_s{seed + 1}.npz"); t0 = time.time()
            torch.save(train(MLP(), SynthSampler(z["Y"], z["L"]), 4000, 1e-3, 64, seed=seed).state_dict(), pp); log(f"mlp pretrain s{seed} {time.time()-t0:.0f}s")
        for k in (4, 8, 0):
            for method in ("scratch", "ft"):
                key = f"{method}|k{k}|s{seed}"
                if key in res: continue
                P = np.full((NS, NT), np.nan); t0 = time.time()
                for f in range(5):
                    test_s = np.where(FOLD == f)[0]; train_all = np.where(FOLD != f)[0]
                    rng = np.random.default_rng(10000 * seed + 100 * f + k)
                    train_s = train_all if k == 0 else rng.choice(train_all, k, replace=False); rs = RealSampler(R, train_s); sd = 7919 * seed + f
                    if method == "scratch": net = train(MLP(), rs, 1500, 1e-3, 64, seed=sd)
                    else:
                        net = MLP(); net.load_state_dict(torch.load(pp)); net = train(net, rs, 500, 3e-4, 64, seed=sd)
                    pr = predict_samples(net, R, test_s, seed=sd)
                    for s in test_s: P[s] = pr[s]
                res[key] = P; pickle.dump(res, open(path, "wb")); log(f"mlp {key} macro {metrics(P, Y)['macro_rmse']:.3f} {time.time()-t0:.0f}s")


{"sal": part_sal, "xmat": part_xmat, "mlp": part_mlp}[a.part]()
log(f"DONE {a.part}")
