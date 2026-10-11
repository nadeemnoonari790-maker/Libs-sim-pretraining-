"""Control dataset: identical synthetic spectra, labels randomly permuted across spectra (breaks the spectrum->concentration link)."""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, numpy as np
CACHE = _R + "cache/"
for s in (1, 2, 3):
    try:
        z = np.load(CACHE + f"synth_informed_s{s}.npz")
    except FileNotFoundError:
        continue
    rng = np.random.default_rng(777 + s)
    L = z["L"][rng.permutation(len(z["L"]))]
    np.savez_compressed(CACHE + f"synth_shuf_s{s}.npz", Y=z["Y"], L=L, P=z["P"], targets=z["targets"], cfg="shuffled-labels control")
    print("wrote shuf", s, L.shape)
