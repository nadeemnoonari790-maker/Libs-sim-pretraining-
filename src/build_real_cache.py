"""Build a cleaned cache of REAL laboratory LIBS spectra (ChemiCam open database).

Source: ocean-perception/chemicam_open_database (Takahashi et al., Deep-Sea Res. I, 2020).
Only files recorded on/after 2015-12-12 are used: they share one wavelength calibration
(ChemiCamF_wavelength_20151212.txt, 321.3-559.3 nm, 1024 px). Earlier files use a different
range and show wavelength drift of up to ~0.8 nm against known Cu/Zn lines, so they are excluded.

Per file: read in chunks, keep the last 1024 columns (spectral pixels), average consecutive
blocks of BLOCK shots (per-block mean raw counts). Shots with a non-finite value or an
anomalously low integrated signal (laser missed / pellet edge) are dropped before averaging.
No normalisation is applied here; that is done identically for real and synthetic spectra later.
"""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import os, re, sys, json
import numpy as np
import pandas as pd

ROOT = _CHEMICAM + ""
LAB = ROOT + "spectraDB/Laboratory/"
OUT = _R + "cache/real_blocks.npz"
BLOCK = 20
MIN_DATE = 20151212
NPIX = 1024


def parse_comp(name):
    p = f"{ROOT}sampleDB/{name}.txt"
    if not os.path.exists(p):
        return None
    d = {}
    for tok in open(p).read().strip().split(";"):
        if "," in tok:
            k, v = tok.split(",", 1)
            try:
                d[k] = float(v)
            except ValueError:
                pass
    return d


def iter_blocks(path):
    """Yield (block_mean_spectrum, n_shots_used) for one csv file."""
    buf = []
    for chunk in pd.read_csv(path, header=None, dtype=np.float64, engine="c", chunksize=2000,
                             on_bad_lines="skip"):
        S = chunk.iloc[:, -NPIX:].to_numpy()
        ok = np.isfinite(S).all(axis=1)
        buf.append(S[ok])
    if not buf:
        return
    S = np.concatenate(buf, axis=0)
    off = np.percentile(S, 2, axis=1)
    tot = (S - off[:, None]).sum(axis=1)
    # drop shots whose integrated signal is far below the file median (missed / edge shots)
    keep = tot > 0.3 * np.median(tot)
    S = S[keep]
    nb = len(S) // BLOCK
    for b in range(nb):
        yield S[b * BLOCK:(b + 1) * BLOCK].mean(axis=0), BLOCK


def main():
    X, sid, session, fidx = [], [], [], []
    names, meta = [], {}
    for s in sorted(os.listdir(LAB)):
        d = LAB + s
        if not os.path.isdir(d):
            continue
        comp = parse_comp(s)
        if comp is None:
            continue
        files = [f for f in sorted(os.listdir(d)) if f.endswith(".csv")
                 and int(re.search(r"_(\d{8})_", f).group(1)) >= MIN_DATE]
        if not files:
            continue
        if s not in names:
            names.append(s)
            meta[s] = comp
        n0 = len(X)
        for f in files:
            date = int(re.search(r"_(\d{8})_", f).group(1))
            for spec, _ in iter_blocks(f"{d}/{f}"):
                X.append(spec.astype(np.float32))
                sid.append(names.index(s))
                session.append(date // 10000)
                fidx.append(f)
        print(f"{s:<20} files={len(files)} blocks={len(X) - n0}", flush=True)
    X = np.stack(X)
    wl = np.loadtxt(ROOT + "calibrationDB/ChemiCamF_wavelength_20151212.txt")
    np.savez_compressed(OUT, X=X, sample_idx=np.array(sid), year=np.array(session),
                        source_file=np.array(fidx), names=np.array(names), wl=wl)
    json.dump(meta, open(OUT.replace(".npz", "_composition.json"), "w"))
    print("saved", X.shape, "samples", len(names))


if __name__ == "__main__":
    main()
