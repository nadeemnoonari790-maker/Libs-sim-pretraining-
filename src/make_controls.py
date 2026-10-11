"""Extra control / scaling datasets derived from the existing informed sets.
elperm_s*: identical spectra, labels' ELEMENT columns permuted by a fixed derangement (spectral physics intact, line->element mapping wrong).
n1500_s1, n6000_s1: first N spectra of informed_s1 (pre-training set-size scaling)."""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import numpy as np
CACHE = _R + "cache/"
PERM = np.array([(i + 3) % 11 for i in range(11)])          # derangement: no element keeps its own label
assert len(set(PERM)) == 11 and not np.any(PERM == np.arange(11))
for s in (1, 2, 3):
    z = np.load(CACHE + f"synth_informed_s{s}.npz")
    np.savez_compressed(CACHE + f"synth_elperm_s{s}.npz", Y=z["Y"], L=z["L"][:, PERM], P=z["P"], targets=z["targets"], cfg="element-permuted labels control")
    print("elperm", s)
z = np.load(CACHE + "synth_informed_s1.npz")
for n in (1500, 6000):
    np.savez_compressed(CACHE + f"synth_n{n}_s1.npz", Y=z["Y"][:n], L=z["L"][:n], P=z["P"][:n], targets=z["targets"], cfg=f"first {n} spectra")
    print("subset", n)
