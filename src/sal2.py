"""Dataset 2: input-gradient attribution specificity (same definition as dataset 1, k = all training targets).
For each element e: share of |grad x input| falling on e's own strongest line windows (not shared with other scored
elements' windows), normalised by window width, divided by the same quantity for the other elements' windows.
"""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, os, time, pickle, sqlite3, json
sys.path.insert(0, _SRC)
import numpy as np, torch
from data import make_folds
from data2 import load_real2, TARGETS2, CACHE2, WL2, EDGES
from models import Net, train, SCALE
from exp2 import RealSampler2, NT2
from synth2 import instrument_info

RES = _R + "results/"
R = load_real2(); FOLD = make_folds(R["groups"])
MASK = instrument_info()[0] > 0
HALF = np.concatenate([np.diff(e) / 2 for e in EDGES])          # half pixel width


def windows():
    con = sqlite3.connect("file:" + _R + "data/atomic_lines.db?mode=ro", uri=True)
    W = {}
    for e in TARGETS2:
        r = np.array(con.execute("select wavelength_nm,aki,gk,ek_ev from lines where element=? and sp_num in (1,2) and "
                                 "wavelength_nm between ? and ? and aki>0 and gk>0 and ek_ev<12", (e, WL2[0], WL2[-1])).fetchall(), float)
        # keep lines that land on reported (non-masked) pixels
        pix = np.clip(np.searchsorted(WL2, r[:, 0]), 0, len(WL2) - 1)
        r = r[MASK[pix]]
        w = r[:, 1] * r[:, 2] * np.exp(-r[:, 3] / 1.0)
        top = r[np.argsort(-w)[:12], 0]
        m = np.zeros(len(WL2), bool)
        for l in top:
            m |= np.abs(WL2 - l) <= HALF + 0.15
        W[e] = m
    return W


def saliency(net, test_s):
    net.eval()
    idx = np.concatenate([R["start"][s] + np.arange(R["count"][s]) for s in test_s])
    x = torch.from_numpy((R["Y"][idx] * SCALE).astype(np.float32))[:, None, :].requires_grad_(True)
    out = net(x); S = np.zeros((NT2, len(WL2)))
    for j in range(NT2):
        g, = torch.autograd.grad(out[:, j].sum(), x, retain_graph=True)
        S[j] = (g * x).abs().squeeze(1).mean(0).detach().numpy()
    return S


def pre(tag, seed):
    n = Net(NT2); n.load_state_dict(torch.load(CACHE2 + f"pre_{tag}_s{seed + 1}.pt")); return n


def run():
    path = RES + "sal2.pkl"; res = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    for seed in (0, 1, 2):
        for f in range(5):
            key = f"s{seed}|f{f}"
            if key in res:
                continue
            test_s = np.where(FOLD == f)[0]; train_s = np.where(FOLD != f)[0]; rs = RealSampler2(R, train_s); sd = 7919 * seed + f
            t0 = time.time(); o = {}
            o["scratch"] = saliency(train(Net(NT2), rs, 1500, 1e-3, 64, seed=sd), test_s)
            o["pretrained"] = saliency(train(pre("synth2", seed), rs, 500, 3e-4, 64, seed=sd), test_s)
            o["elperm"] = saliency(train(pre("elperm2", seed), rs, 500, 3e-4, 64, seed=sd), test_s)
            o["pre_only"] = saliency(pre("synth2", seed), test_s)
            res[key] = o; pickle.dump(res, open(path, "wb")); print(f"sal2 {key} {time.time()-t0:.0f}s", flush=True)


def analyse():
    from scipy.stats import wilcoxon
    sal = pickle.load(open(RES + "sal2.pkl", "rb")); W = windows()

    def spec(S, e):
        j = TARGETS2.index(e); own = W[e]; oth = np.zeros(len(WL2), bool)
        for e2 in TARGETS2:
            if e2 != e:
                oth |= W[e2]
        o1, o2 = own & ~oth, oth & ~own
        if o1.sum() == 0:
            return np.nan
        s = S[j] / S[j].sum()
        return (s[o1].sum() / o1.mean()) / (s[o2].sum() / o2.mean())
    out = {}
    for m in ("scratch", "pretrained", "elperm", "pre_only"):
        per = {e: float(np.nanmean([spec(d[m], e) for d in sal.values()])) for e in TARGETS2}
        out[m] = dict(per_element=per, macro=float(np.nanmean(list(per.values()))), n=len(sal))
        print(m, "macro %.2f" % out[m]["macro"], {k: round(v, 2) for k, v in per.items()})

    def vec(m):
        return np.array([[spec(d[m], e) for e in TARGETS2] for d in sal.values()]).mean(0)
    for b in ("scratch", "elperm"):
        d = vec("pretrained") - vec(b); d = d[np.isfinite(d)]
        out[f"pretrained_vs_{b}"] = dict(frac_elements_higher=float(np.mean(d > 0)), p=float(wilcoxon(d).pvalue), n=int(len(d)))
        print("pretrained vs", b, out[f"pretrained_vs_{b}"])
    out["window_pixels"] = {e: int(W[e].sum()) for e in TARGETS2}
    json.dump(out, open(RES + "sal2.json", "w"), indent=1)


if __name__ == "__main__":
    {"run": run, "analyse": analyse}[sys.argv[1]]()
