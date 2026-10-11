"""Domain-randomised synthetic LIBS spectrum generator built on the LTE slab forward model.

Every synthetic spectrum = forward model (Saha-Boltzmann, per-line Stark widths, self-absorption, optional cool
absorbing layer) with randomly drawn
  * composition       : broad log-uniform prior over the labelled TARGET elements (+ unlabelled nuisance
                        elements and ambient seawater Ca/Mg that are NOT part of the label),
  * plasma state      : T, ne, column density, Stark scale/exponent,
  * instrument        : Gaussian resolution, wavelength shift/stretch, spectral sensitivity curve,
  * acquisition       : additive offset + smooth continuum, Gaussian noise whose noise/peak ratio is drawn from the
                        EMPIRICAL distribution measured on the real spectra (label-free), occasional hot pixels,
and is then passed through exactly the same preprocessing as the real spectra.

prior = "informed": instrument nuisance parameters centred on values obtained from fitting the brass standards
                    (wavelength offset, sensitivity slope/curvature) - a form of instrument calibration.
prior = "broad"   : the same parameters drawn from wide uninformative ranges (no instrument knowledge).
Concentration labels of the real data are never used to build the generator.
"""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, os
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, _SRC)
from forward_model import AtomicData, LIBSForward
from preprocess import preprocess

CACHE = _R + "cache/"
DB = _R + "data/atomic_lines.db"
REAL = np.load(CACHE + "real_blocks.npz")
WL = REAL["wl"]
U = (WL - 440.0) / 110.0

TARGETS = ["Cu", "Zn", "Fe", "Pb", "Ca", "Mg", "Mn", "Al", "Si", "Ti", "Ba"]
NUIS = ["Ag", "Cd", "Sr", "Na", "K", "Sb", "Ni", "Co", "Cr", "V", "Mo", "Sn"]
ALL_EL = TARGETS + NUIS
NT = len(TARGETS)
# prior upper bound of log10(wt%) per target element
LOGMAX = dict(Cu=2.0, Zn=1.8, Fe=1.75, Pb=1.5, Ca=1.5, Mg=1.4, Mn=1.4, Al=1.1, Si=1.6, Ti=0.8, Ba=1.5)
LOGMIN = -2.0  # 0.01 wt% = label floor

_FM = None


def _init():
    global _FM
    at = AtomicData(DB, ALL_EL, WL[0] - 8, WL[-1] + 8)
    _FM = LIBSForward(at, WL)


def noise_ratio_pool():
    p = CACHE + "noise_ratio.npy"
    if os.path.exists(p):
        return np.load(p)
    X = REAL["X"][::10]
    Y, b = preprocess(X, return_baseline=True)
    hf = np.diff(X.astype(float), axis=1) / np.sqrt(2)
    noise = 1.4826 * np.median(np.abs(hf - np.median(hf, axis=1, keepdims=True)), axis=1)
    peak = (X - b).max(axis=1)
    r = noise / np.maximum(peak, 1e-9)
    np.save(p, r)
    return r


def draw_one(rng, cfg, nr_pool):
    fm = _FM
    informed = cfg.get("prior", "informed") == "informed"
    # ---------- composition (wt% of whole sample; unmodelled matrix such as O, S, C is invisible) ----------
    p_present = rng.uniform(0.25, 0.9)
    c = np.zeros(NT)
    for i, el in enumerate(TARGETS):
        if rng.random() < p_present:
            c[i] = 10 ** rng.uniform(LOGMIN, LOGMAX[el])
    if c.sum() == 0:
        i = rng.integers(NT)
        c[i] = 10 ** rng.uniform(0.0, LOGMAX[TARGETS[i]])
    cap = rng.uniform(40, 100)
    if c.sum() > cap:
        c *= cap / c.sum()
    w = np.zeros(len(ALL_EL))
    w[:NT] = c / 100.0
    if cfg.get("nuisance", True):
        for j in range(len(NUIS)):
            if rng.random() < 0.15:
                w[NT + j] = 10 ** rng.uniform(-3.0, 0.3) / 100.0
    if cfg.get("ambient", True):                       # seawater Ca / Mg lines, not part of the label
        w[TARGETS.index("Ca")] += 10 ** rng.uniform(-4.7, -2.3)
        w[TARGETS.index("Mg")] += 10 ** rng.uniform(-4.7, -2.3)
    w = np.maximum(w, 1e-12)
    # ---------- plasma ----------
    T = rng.uniform(5500, 10500)
    lne = rng.uniform(16.6, 17.9)
    lNL = rng.uniform(14.3, 17.3)
    fg = rng.uniform(0.7, 1.6) if informed else rng.uniform(0.3, 2.0)
    sref = rng.uniform(0.15, 1.0)
    sp = rng.uniform(1.0, 3.0)
    layer2 = None
    if rng.random() < cfg.get("p_layer2", 0.25):
        layer2 = (rng.uniform(3500, 0.85 * T), 10 ** rng.uniform(15.5, 17.0), 10 ** rng.uniform(13.0, 16.5))
    I, _ = fm.spectrum_fine_stark(w, T, 10 ** lne, 10 ** lNL, fg, sref, sp, layer2)
    # ---------- instrument ----------
    if informed:
        sh, st = rng.normal(-0.47, 0.2), rng.normal(0.001, 0.0008)
        s1, s2 = rng.normal(1.0, 0.4), rng.normal(-1.45, 0.35)
    else:
        sh, st = rng.uniform(-1.2, 1.2), rng.uniform(-3e-3, 3e-3)
        s1, s2 = rng.uniform(-3, 3), rng.uniform(-3, 3)
    wl_eff = 440.0 + (fm.wl_pix - 440.0) * (1.0 + st) + sh
    S = np.interp(wl_eff, fm.fine, I) * (WL / 500.0) * np.exp(s1 * U + s2 * U ** 2)
    S = S / max(S.max(), 1e-300)
    # ---------- acquisition ----------
    amp = 10 ** rng.uniform(3.0, 5.0)
    S = S * amp
    cont = amp * rng.uniform(0.0, 0.4) * (1 + rng.normal(0, 0.3) * U + rng.normal(0, 0.3) * U ** 2)
    S = S + 1.5e5 + np.maximum(cont, 0)
    if cfg.get("noise", "empirical") == "empirical":
        r = rng.choice(nr_pool)
        S = S + rng.normal(0, r * amp, S.shape)
        if rng.random() < 0.1:
            S[rng.integers(0, len(S), rng.integers(1, 4))] += amp * rng.uniform(0.2, 1.0)
    lab = c.copy()
    par = np.array([T, lne, lNL, fg, sref, sp, sh, st, s1, s2, amp, float(layer2 is not None)])
    return S.astype(np.float32), lab.astype(np.float32), par.astype(np.float32)


def _work(args):
    seed, n, cfg = args
    rng = np.random.default_rng(seed)
    pool = noise_ratio_pool()
    S, L, P = [], [], []
    for _ in range(n):
        s, l, p = draw_one(rng, cfg, pool)
        S.append(s); L.append(l); P.append(p)
    return np.stack(S), np.stack(L), np.stack(P)


def generate(n, seed, cfg, procs=2, chunk=500):
    jobs = [(seed * 100003 + k, min(chunk, n - k * chunk), cfg) for k in range((n + chunk - 1) // chunk)]
    with Pool(procs, initializer=_init) as pool:
        out = pool.map(_work, jobs)
    S = np.concatenate([o[0] for o in out]); L = np.concatenate([o[1] for o in out]); P = np.concatenate([o[2] for o in out])
    Y = preprocess(S)
    return Y, L, P


if __name__ == "__main__":
    import argparse, time, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="informed")
    ap.add_argument("--prior", default="informed")
    ap.add_argument("--noise", default="empirical")
    ap.add_argument("--ambient", type=int, default=1)
    ap.add_argument("--nuisance", type=int, default=1)
    a = ap.parse_args()
    cfg = dict(prior=a.prior, noise=a.noise, ambient=bool(a.ambient), nuisance=bool(a.nuisance))
    noise_ratio_pool()
    t = time.time()
    Y, L, P = generate(a.n, a.seed, cfg)
    np.savez_compressed(CACHE + f"synth_{a.tag}_s{a.seed}.npz", Y=Y, L=L, P=P, targets=np.array(TARGETS), cfg=json.dumps(cfg))
    print(f"{a.tag} seed={a.seed} n={a.n} Y{Y.shape} finite={np.isfinite(Y).all()} in {time.time()-t:.0f}s cfg={cfg}")
