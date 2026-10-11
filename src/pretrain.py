"""Pretrain the CNN on synthetic spectra only; report zero-shot accuracy on ALL real samples (monitoring, never used for tuning)."""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, json, time, os, argparse
sys.path.insert(0, _SRC)
import numpy as np, torch
from data import *
from models import *

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="informed"); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
a = ap.parse_args()

R = load_real()
z = np.load(CACHE + f"synth_{a.tag}_s{a.seed}.npz")
S = SynthSampler(z["Y"], z["L"])
t0 = time.time()
net = Net()
log = lambda s: print(s, flush=True)
train(net, S, a.steps, a.lr, a.bs, seed=a.seed, log=log)
torch.save(net.state_dict(), CACHE + f"pre_{a.tag}_s{a.seed}{CROP_SUFFIX}.pt")
pr = predict_samples(net, R, range(len(R["names"])), seed=0)
P = np.stack([pr[s] for s in range(len(R["names"]))])
m = metrics(P, R["Ylog"])
os.makedirs(_R + "results", exist_ok=True)
fn = _R + "results/zeroshot.json"
d = json.load(open(fn)) if os.path.exists(fn) else {}
d[f"{a.tag}_s{a.seed}{CROP_SUFFIX}"] = dict(metrics=m, train_s=time.time() - t0, steps=a.steps)
json.dump(d, open(fn, "w"), indent=1)
np.save(CACHE + f"zs_pred_{a.tag}_s{a.seed}{CROP_SUFFIX}.npy", P)
print(f"[{a.tag} s{a.seed}] zero-shot macro RMSE {m['macro_rmse']:.3f}  MAE {m['macro_mae']:.3f}  R2 {m['macro_r2']:.3f}  "
      f"({time.time()-t0:.0f}s)  per-element: " + " ".join(f"{e}={m[e]:.2f}" for e in SCORED))
