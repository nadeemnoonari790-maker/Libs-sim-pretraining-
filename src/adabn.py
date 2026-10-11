"""Label-free adaptation probe: re-estimate BatchNorm statistics of the synthetic-only network on UNLABELLED real training-fold spectra (AdaBN),
then predict held-out samples zero-shot (no labels used). Same folds as the main experiment -> no test-spectra leakage."""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, json
sys.path.insert(0, _SRC)
import numpy as np, torch
from data import *
from models import *

R = load_real(); FOLD = make_folds(R["groups"]); Y = R["Ylog"]; NS = len(R["names"])
out = {}
for seed in (1, 2, 3):
    base = Net(); base.load_state_dict(torch.load(CACHE + f"pre_informed_s{seed}.pt"))
    pred = np.full((NS, len(TARGETS)), np.nan); rng = np.random.default_rng(seed)
    for f in range(5):
        test_s = np.where(FOLD == f)[0]; train_s = np.where(FOLD != f)[0]
        net = Net(); net.load_state_dict(base.state_dict())
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.reset_running_stats(); m.momentum = None          # cumulative average
        net.train()                                                  # BN uses batch stats and accumulates running stats
        rs = RealSampler(R, train_s)
        with torch.no_grad():
            for _ in range(40):
                x, _, _ = rs(rng, 128); net(torch.from_numpy(x.astype(np.float32))[:, None, :])
        net.eval()
        pr = predict_samples(net, R, test_s, seed=seed)
        for s in test_s: pred[s] = pr[s]
    m = metrics(pred, Y); out[f"informed_s{seed}"] = m
    zs = json.load(open(_R + "results/zeroshot.json"))[f"informed_s{seed}"]["metrics"]
    print(f"seed {seed}: zero-shot {zs['macro_rmse']:.3f} -> AdaBN {m['macro_rmse']:.3f}  per-element:", {e: round(m[e], 2) for e in SCORED}, flush=True)
json.dump(out, open(_R + "results/adabn.json", "w"), indent=1)
print("mean AdaBN macro RMSE", np.mean([v["macro_rmse"] for v in out.values()]))
