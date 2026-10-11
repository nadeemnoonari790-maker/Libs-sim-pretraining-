"""Dataset 2: Mount Holyoke ChemLIBS reference database (NASA PDS, doi:10.17189/b2aj-cz96), Mars atmosphere (7 Torr CO2),
5 mJ attenuation setting, locations 1-2 of every target (each file = mean of 5 integrations x 10 shots).

Protocol (fixed before any model was evaluated on this dataset):
  pixels : three spectrometer segments re-binned to 1024 pixels (UV 241.0-340.5 nm and VIS 382.25-469.0 nm at 0.25 nm/px,
           VNIR 473.5-904.5 nm at ~1.54 nm/px) by exact bin averaging; the SAME binning is applied to synthetic spectra.
  labels : element wt% computed stoichiometrically from the certified/XRF oxide wt% (Fe2O3* = total Fe);
           log10(max(wt%, 0.01)); missing values masked.
  filter : targets whose 10-oxide total is 85-105 wt% and whose 9 major oxides (all but P2O5) are reported
           (removes pure metals, Ti alignment standard, incomplete analyses).
  groups : targets sharing an alphabetic name prefix (sample suite / collection) are held out together.
  scored : elements with >= 15 targets at >= 0.1 wt% spread over >= 10 groups.
"""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, _SRC)
from preprocess import preprocess

ROOT = _R + ""
CACHE2 = ROOT + "cache/chemlibs/"
os.makedirs(CACHE2, exist_ok=True)
TARGETS2 = ["Si", "Ti", "Al", "Fe", "Mn", "Mg", "Ca", "Na", "K", "P"]
OXIDES = ["SiO2", "TiO2", "Al2O3", "Fe2O3*", "MnO", "MgO", "CaO", "Na2O", "K2O", "P2O5"]
# mass fraction of the element in its oxide
FACT = dict(Si=28.0855 / 60.0843, Ti=47.867 / 79.866, Al=2 * 26.9815 / 101.961, Fe=2 * 55.845 / 159.688,
            Mn=54.938 / 70.937, Mg=24.305 / 40.304, Ca=40.078 / 56.077, Na=2 * 22.9898 / 61.979,
            K=2 * 39.0983 / 94.196, P=2 * 30.9738 / 141.945)
FLOOR = 0.01
NPIX = 1024
SEG = [(241.0, 340.5, 0.25), (382.25, 469.0, 0.25)]


def pixel_edges():
    e = [np.arange(a, b + 1e-9, s) for a, b, s in SEG]
    n_used = sum(len(x) - 1 for x in e)
    e.append(np.linspace(473.5, 904.5, NPIX - n_used + 1))
    return e


EDGES = pixel_edges()
WL2 = np.concatenate([(x[:-1] + x[1:]) / 2 for x in EDGES])
assert len(WL2) == NPIX


def bin_avg(grid, Y):
    """Exact average of piecewise-linear spectra Y(grid) over every pixel bin. Y: (n, len(grid))."""
    Y = np.atleast_2d(np.asarray(Y, np.float64))
    cum = np.concatenate([np.zeros((len(Y), 1)), np.cumsum((Y[:, 1:] + Y[:, :-1]) / 2 * np.diff(grid)[None, :], axis=1)], 1)
    out = []
    for e in EDGES:
        ce = np.stack([np.interp(e, grid, c) for c in cum])
        out.append(np.diff(ce, axis=1) / np.diff(e)[None, :])
    return np.concatenate(out, axis=1).astype(np.float32)


def group_of(n):
    m = re.match(r"^([A-Z]+)", n)
    if m:
        return m.group(1)
    m = re.match(r"^\d+([A-Z]+)$", n)
    return "N" + (m.group(1) if m else "")


def to_log(c):
    return np.log10(np.maximum(c, FLOOR))


def build():
    d = np.load(ROOT + "data/chemlibs/chemlibs_mars5mj.npz")
    grid = d["grid"].astype(np.float64); S = d["spec"].astype(np.float64); files = [str(x) for x in d["names"]]
    fin = np.isfinite(S).all(axis=0)
    tgt = [re.search(r"_t\d+l(\d+)_(.+)_spect\.csv$", f, re.I).group(2).upper() for f in files]
    meta = pd.read_excel(ROOT + "data/chemlibs/libs_metadata.xlsx").iloc[1:].reset_index(drop=True)
    meta["N"] = meta["PELLET NAME"].astype(str).str.upper().str.strip()
    O = meta[OXIDES].apply(pd.to_numeric, errors="coerce")
    tot = O.sum(axis=1, min_count=8)
    ok = (tot >= 85) & (tot <= 105) & O[OXIDES[:9]].notna().all(axis=1)
    comp = {n: np.array([O.loc[i, ox] * FACT[e] for e, ox in zip(TARGETS2, OXIDES)], float)
            for i, n in enumerate(meta["N"]) if ok[i]}
    # bin each segment on its own finite grid (no interpolation across spectrometer gaps)
    X = bin_avg(grid[fin], S[:, fin])
    keep = np.array([t in comp for t in tgt])
    X, tgt = X[keep], [t for t, k in zip(tgt, keep) if k]
    names = sorted(set(tgt))
    sid = np.array([names.index(t) for t in tgt]); o = np.argsort(sid, kind="stable")
    X, sid = X[o], sid[o]
    C = np.stack([comp[n] for n in names])
    np.savez_compressed(CACHE2 + "real2.npz", X=X, sid=sid, names=np.array(names), C=C, wl=WL2)
    print("built", X.shape, "targets", len(names), "dropped spectra", int((~keep).sum()))


def load_real2():
    p = CACHE2 + "real2.npz"
    if not os.path.exists(p):
        build()
    D = np.load(p)
    pp = CACHE2 + "real2_pre.npy"
    if os.path.exists(pp):
        Y = np.load(pp)
    else:
        Y = preprocess(D["X"]); np.save(pp, Y)
    names = [str(n) for n in D["names"]]; sid = D["sid"]; C = D["C"]
    start = np.searchsorted(sid, np.arange(len(names))); count = np.bincount(sid, minlength=len(names))
    Ylog = np.where(np.isnan(C), np.nan, to_log(np.nan_to_num(C)))
    groups = np.array([group_of(n) for n in names])
    return dict(Y=Y, X=D["X"], sid=sid, start=start, count=count, names=names, C=C, Ylog=Ylog, groups=groups, wl=D["wl"])


def scored_elements(R):
    out = []
    for j, e in enumerate(TARGETS2):
        v = R["C"][:, j] >= 0.1
        if v.sum() >= 15 and len(set(R["groups"][v])) >= 10:
            out.append(e)
    return out


if __name__ == "__main__":
    R = load_real2()
    print("spectra", R["Y"].shape, "targets", len(R["names"]), "groups", len(set(R["groups"])))
    print("scored", scored_elements(R))
    for j, e in enumerate(TARGETS2):
        c = R["C"][:, j]; print(e, "n", int(np.isfinite(c).sum()), "median %.3f" % np.nanmedian(c), "max %.2f" % np.nanmax(c),
                               ">=0.1:", int((c >= 0.1).sum()))
