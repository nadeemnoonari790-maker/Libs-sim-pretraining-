"""Domain-randomised synthetic generator for dataset 2 (ChemLIBS, Mars atmosphere). Same LTE slab forward model as dataset 1.

Only label-free, published instrument information is used: the three spectrometer ranges and nominal resolutions
(user guide: 0.14 nm UV/VIS, 0.55 nm VNIR), the instrument's masked (zero) pixels, and the empirical noise/peak ratio of
the real spectra. No real concentration label is used and no instrument parameter is fitted to real spectra
(this corresponds to the "broad" prior of dataset 1).

composition : rock-like prior; labelled elements Si..P log-uniform, O = matrix (unlabelled), ambient C/O from CO2
              (unlabelled), occasional unlabelled trace/nuisance elements.
plasma      : T, ne, column density, per-line Stark widths, optional cooler absorbing layer.
instrument  : Gaussian resolution per spectrometer, small wavelength shift/stretch, smooth residual response,
              independent gain per spectrometer, continuum, offset, empirical noise.
"""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, os, json, time, argparse
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, _SRC)
from forward_model import AtomicData, LIBSForward
from preprocess import preprocess
from data2 import WL2, EDGES, bin_avg, CACHE2, TARGETS2, load_real2

DB = _R + "data/atomic_lines.db"
NUIS2 = ["Sr", "Ba", "Cr", "Ni", "Zn", "Cu", "Li", "Rb", "Co", "V", "S"]
ALL2 = TARGETS2 + ["O", "C"] + NUIS2
NT2 = len(TARGETS2)
LOGMAX2 = dict(Si=1.68, Ti=0.8, Al=1.42, Fe=1.70, Mn=0.8, Mg=1.48, Ca=1.60, Na=0.95, K=1.05, P=0.5)
LOGMIN2 = -2.0
U2 = (WL2 - 570.0) / 330.0
SEGID = np.concatenate([np.full(len(e) - 1, k) for k, e in enumerate(EDGES)])
_FM = None


def instrument_info():
    p = CACHE2 + "instr.npz"
    if os.path.exists(p):
        z = np.load(p); return z["mask"], z["nr"]
    R = load_real2()
    X = R["X"].astype(np.float64)
    mask = (np.median(X, axis=0) > 0).astype(np.float32)          # pixels the instrument reports (non-masked)
    Y, b = preprocess(X[::4], return_baseline=True)
    Xs = X[::4]
    hf = np.diff(Xs, axis=1) / np.sqrt(2)
    good = (mask[1:] * mask[:-1] * (SEGID[1:] == SEGID[:-1])) > 0
    hf = hf[:, good]
    noise = 1.4826 * np.median(np.abs(hf - np.median(hf, axis=1, keepdims=True)), axis=1)
    nr = noise / np.maximum((Xs - b).max(axis=1), 1e-9)
    np.savez(p, mask=mask, nr=nr)
    return mask, nr


def _init():
    global _FM
    at = AtomicData(DB, ALL2, WL2[0] - 8, WL2[-1] + 8)
    _FM = LIBSForward(at, WL2)


def draw_one(rng, mask, nr_pool, gainc=(1.0, 1.0, 1.0)):
    fm = _FM
    # ---------- composition (element wt%) ----------
    c = np.zeros(NT2)
    p_present = rng.uniform(0.6, 1.0)
    for i, el in enumerate(TARGETS2):
        if rng.random() < p_present:
            c[i] = 10 ** rng.uniform(LOGMIN2, LOGMAX2[el])
    cap = rng.uniform(30, 65)
    if c.sum() > cap:
        c *= cap / c.sum()
    w = np.zeros(len(ALL2))
    w[:NT2] = c / 100.0
    w[NT2] = (100.0 - c.sum()) * rng.uniform(0.6, 1.0) / 100.0          # O (matrix, unlabelled)
    w[NT2 + 1] = 10 ** rng.uniform(-3.5, -1.0) / 100.0                    # C (CO2 ambient, unlabelled)
    w[NT2] += 10 ** rng.uniform(-3.5, -1.0) / 100.0                       # O from CO2
    for j in range(len(NUIS2)):
        if rng.random() < 0.2:
            w[NT2 + 2 + j] = 10 ** rng.uniform(-3.0, -0.5) / 100.0
    w = np.maximum(w, 1e-12)
    # ---------- plasma ----------
    T = rng.uniform(5000, 12000)
    lne = rng.uniform(16.3, 17.7)
    lNL = rng.uniform(14.5, 17.5)
    g_uv = rng.uniform(0.10, 0.35)
    sref = rng.uniform(0.1, 1.0)
    sp = rng.uniform(1.0, 3.0)
    layer2 = None
    if rng.random() < 0.25:
        layer2 = (rng.uniform(3500, 0.85 * T), 10 ** rng.uniform(15.5, 17.0), 10 ** rng.uniform(13.0, 16.5))
    I, _ = fm.spectrum_fine_stark(w, T, 10 ** lne, 10 ** lNL, g_uv, sref, sp, layer2)
    # VNIR spectrometer: coarser resolution (nominal 0.55 nm)
    g_nir = rng.uniform(0.45, 0.9)
    extra = np.sqrt(max(g_nir ** 2 - g_uv ** 2, 1e-6)) / 2.3548 / fm.step
    Inir = gaussian_filter1d(I, extra)
    I = np.where(fm.fine > 471.0, Inir, I)
    I = I * fm.fine / 500.0                                                # energy -> photon units
    # ---------- instrument ----------
    sh, st = rng.normal(0.0, 0.08), rng.normal(0.0, 2e-4)
    wl_src = fm.fine * (1.0 + st) + sh                                     # where each fine sample really lands
    I_obs = np.interp(fm.fine, wl_src, I)
    S = bin_avg(fm.fine, I_obs[None, :])[0].astype(np.float64)
    s1, s2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
    gains = np.asarray(gainc) * 10 ** rng.uniform(-0.3, 0.3, 3)
    S = S * np.exp(s1 * U2 + s2 * U2 ** 2) * gains[SEGID]
    S = S / max(S.max(), 1e-300)
    # ---------- acquisition ----------
    amp = 10 ** rng.uniform(3.0, 5.0)
    S = S * amp
    cont = amp * rng.uniform(0.0, 0.5) * np.maximum(1 + rng.normal(0, 0.4) * U2 + rng.normal(0, 0.4) * U2 ** 2, 0) * gains[SEGID]
    S = S + cont + amp * rng.uniform(0, 0.05)
    r = rng.choice(nr_pool)
    S = S + rng.normal(0, r * amp, S.shape)
    if rng.random() < 0.1:
        S[rng.integers(0, len(S), rng.integers(1, 4))] += amp * rng.uniform(0.2, 1.0)
    S = S * mask
    par = np.array([T, lne, lNL, g_uv, g_nir, sref, sp, sh, st, s1, s2, amp, float(layer2 is not None)])
    return S.astype(np.float32), c.astype(np.float32), par.astype(np.float32)


def seg_frac(Y):
    """Share of (preprocessed) spectral energy falling in each spectrometer segment."""
    E = np.stack([(Y[:, SEGID == k] ** 2).sum(1) for k in range(3)], 1)
    return E / E.sum(1, keepdims=True)


def calibrate_gains(n=400):
    """Label-free: centre the per-spectrometer gain prior so that the median segment energy share of synthetic spectra
    matches that of the (unlabelled) real spectra. Iterated twice (the share is a non-linear function of the gains)."""
    p = CACHE2 + "gainc.npy"
    if os.path.exists(p):
        return np.load(p)
    mask, nr = instrument_info(); _init()
    fr = np.median(seg_frac(load_real2()["Y"]), 0)
    g = np.ones(3)
    for it in range(3):
        rng = np.random.default_rng(99 + it)
        Ys = preprocess(np.stack([draw_one(rng, mask, nr, g)[0] for _ in range(n)]))
        fs = np.median(seg_frac(Ys), 0)
        g = g * np.sqrt(fr / fs); g = g / g[0]
        print("gain calib iter", it, "real share", fr.round(3), "synth share", fs.round(3), "-> gains", g.round(3), flush=True)
    np.save(p, g)
    return g


def _work(args):
    seed, n = args
    rng = np.random.default_rng(seed)
    mask, nr = instrument_info()
    g = np.load(CACHE2 + "gainc.npy")
    out = [draw_one(rng, mask, nr, g) for _ in range(n)]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out]), np.stack([o[2] for o in out])


def generate(n, seed, procs=2, chunk=500):
    jobs = [(seed * 100003 + k, min(chunk, n - k * chunk)) for k in range((n + chunk - 1) // chunk)]
    with Pool(procs, initializer=_init) as pool:
        out = pool.map(_work, jobs)
    S = np.concatenate([o[0] for o in out]); L = np.concatenate([o[1] for o in out]); P = np.concatenate([o[2] for o in out])
    return preprocess(S), L, P, S


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24000); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    instrument_info(); calibrate_gains()
    t = time.time()
    Y, L, P, S = generate(a.n, a.seed)
    np.savez_compressed(CACHE2 + f"synth2_s{a.seed}.npz", Y=Y, L=L, P=P, targets=np.array(TARGETS2))
    if a.seed == 1:
        np.save(CACHE2 + "synth2_raw_examples.npy", S[:200])
    print(f"seed={a.seed} n={a.n} Y{Y.shape} finite={np.isfinite(Y).all()} {time.time()-t:.0f}s", flush=True)
