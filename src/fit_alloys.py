"""Experiment A: fidelity of the forward model against REAL brass-standard spectra.

For each alloy: fit plasma/instrument nuisance parameters (composition fixed to the certified value)
by differential evolution, report Pearson r / relative L2 error on the 330-545 nm window.
Control: refit with a *different* alloy's composition -> agreement must drop if the model is
composition-informative.
"""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, json, time
import numpy as np
from scipy.optimize import differential_evolution
sys.path.insert(0, _SRC)
from forward_model import AtomicData, LIBSForward
from preprocess import preprocess

D = np.load(_R + "cache/real_blocks.npz")
WL, X, NAMES, IDX = D["wl"], D["X"], list(D["names"]), D["sample_idx"]
COMP = json.load(open(_R + "cache/real_blocks_composition.json"))
WIN = (WL >= 330) & (WL <= 545)
EL = ["Cu", "Zn", "Pb", "Fe", "Ca"]
AT = AtomicData(_R + "data/atomic_lines.db", EL, WL[0] - 8, WL[-1] + 8)
FM = LIBSForward(AT, WL)
ALLOYS = ["31X78355", "31X78358", "31XB2", "31XB20", "31XB21", "31XB23", "C1220", "C2600", "C2801", "C3713", "C5191", "C6871", "C7701"]
U = (WL - 440.0) / 110.0
ZONES = 1
BOUNDS1 = [(5000, 14000),    # T [K]
          (16.0, 18.5),      # log10 ne
          (14.0, 18.5),      # log10 NL
          (0.25, 2.2),       # Gaussian FWHM [nm]
          (0.01, 1.5),       # Stark reference
          (-1.5, 1.5),       # wavelength shift at 440 nm [nm]
          (-4e-3, 4e-3),     # wavelength stretch about 440 nm
          (-3.0, 3.0),       # log-sensitivity slope
          (-3.0, 3.0),       # log-sensitivity curvature
          (-6.0, -1.0),      # log10 Ca mass fraction (seawater ambient)
          (0.5, 3.0)]        # Stark exponent p
BOUNDS2 = BOUNDS1 + [(3000, 9000),   # T2 [K] cool layer
                     (15.0, 18.0),   # log10 ne2
                     (12.0, 17.5)]   # log10 NL2


def mass_frac(comp, ca):
    w = np.array([comp["Cu"], comp["Zn"], comp["Pb"], comp["Fe"]], float)
    w = np.maximum(w, 1e-6)
    w = w / w.sum()
    return np.concatenate([w * (1 - ca), [ca]])


def simulate(theta, comp):
    T, lne, lNL, fg, sref, sh, st, s1, s2, lca, sp = theta[:11]
    l2 = None
    if len(theta) > 11:
        l2 = (min(theta[11], T), 10 ** theta[12], 10 ** theta[13])
    I, _ = FM.spectrum_fine_stark(mass_frac(comp, 10 ** lca), T, 10 ** lne, 10 ** lNL, fg, sref, sp, l2)
    wl_eff = 440.0 + (FM.wl_pix - 440.0) * (1.0 + st) + sh
    S = np.interp(wl_eff, FM.fine, I) * (WL / 500.0) * np.exp(s1 * U + s2 * U ** 2)
    return preprocess(S)[0]


def norm_win(y):
    y = y[WIN]
    return y / max(np.linalg.norm(y), 1e-12)


def objective(theta, comp, target):
    return float(np.sum((norm_win(simulate(theta, comp)) - target) ** 2))


def fit(comp, target, seed=0):
    r = differential_evolution(objective, BOUNDS2 if ZONES == 2 else BOUNDS1, args=(comp, target), seed=seed, popsize=14, maxiter=90,
                               tol=1e-7, polish=False, workers=2, updating="deferred", mutation=(0.5, 1.0), recombination=0.8)
    return r.x, r.fun


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--zones", type=int, default=1); ap.add_argument("--alloys", default="all")
    ap.add_argument("--tag", default="v2"); ap.add_argument("--control", type=int, default=1)
    a = ap.parse_args(); ZONES = a.zones
    sel = ALLOYS if a.alloys == "all" else a.alloys.split(",")
    out = {}; t0 = time.time()
    for k, al in enumerate(sel):
        real = preprocess(X[IDX == NAMES.index(al)].mean(axis=0))[0]; target = norm_win(real)
        th, fv = fit(COMP[al], target)
        r_true = pearson(norm_win(simulate(th, COMP[al])), target)
        rec = dict(theta=list(map(float, th)), r_true=r_true, rel_l2_true=float(np.sqrt(fv)))
        if a.control:
            wrong = ALLOYS[(ALLOYS.index(al) + 4) % len(ALLOYS)]
            th2, f2 = fit(COMP[wrong], target, seed=1)
            rec.update(wrong_alloy=wrong, r_wrong=pearson(norm_win(simulate(th2, COMP[wrong])), target))
        out[al] = rec
        print(f"zones={ZONES} {al:<9} r(true)={r_true:.3f}" + (f"  r(wrong:{rec['wrong_alloy']})={rec['r_wrong']:.3f}" if a.control else "") +
              f"  T={th[0]:.0f} lne={th[1]:.2f} lNL={th[2]:.2f} Gfwhm={th[3]:.2f} Stark={th[4]:.2f} p={th[10]:.2f} [{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open(_R + f"results/expA_fit_{a.tag}.json", "w"), indent=1)
