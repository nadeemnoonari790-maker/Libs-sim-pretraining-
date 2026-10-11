"""Physics sanity tests for the forward model. Run: python3 -I src/test_forward_model.py"""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, time
import numpy as np
sys.path.insert(0, _SRC)
from forward_model import AtomicData, LIBSForward, KB_EV

WL = np.loadtxt(_CHEMICAM + "calibrationDB/ChemiCamF_wavelength_20151212.txt")
EL = ["Cu", "Zn", "Pb", "Fe"]
at = AtomicData(_R + "data/atomic_lines.db", EL, WL[0] - 8, WL[-1] + 8)
fm = LIBSForward(at, WL)
print(f"lines in model: {at.n_lines}  fine grid: {len(fm.fine)} pts")
ok = True


def check(name, cond, msg=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + "  " + msg)


# ---- Test 1: Boltzmann-plot temperature recovery (pure Cu, optically thin) ----
T_true, ne = 9000.0, 1e17
I_thin, tau = fm.thin_spectrum_fine([1, 0, 0, 0], T_true, ne, 1e11, 0.25, 0.02)   # tiny NL -> thin
cu = np.where((at.elem_of_sp[at.sp_idx] == 0) & (at.sp_idx == 0))[0]  # Cu I lines
ln_y, Ek = [], []
for i in cu:
    w0 = at.wl[i]
    if not (fm.fine[0] + 4 < w0 < fm.fine[-1] - 4):
        continue
    # skip lines with a neighbour within 1.2 nm (blends)
    if np.sum(np.abs(at.wl[cu] - w0) < 1.2) > 1:
        continue
    m = (fm.fine > w0 - 1.5) & (fm.fine < w0 + 1.5)
    area = I_thin[m].sum() * fm.step
    if area <= 0:
        continue
    ln_y.append(np.log(area * w0 / (at.aki[i] * at.gk[i])))
    Ek.append(at.ek[i])
ln_y, Ek = np.array(ln_y), np.array(Ek)
slope, icpt = np.polyfit(Ek, ln_y, 1)
T_fit = -1.0 / (slope * KB_EV)
check("Boltzmann plot recovers T (thin Cu I)", abs(T_fit - T_true) / T_true < 0.03,
      f"T_true={T_true:.0f} K  T_fit={T_fit:.0f} K  ({len(Ek)} isolated lines)")

# ---- Test 2: self-absorption of the Cu I 324.75/327.40 resonance doublet ----
def doublet_ratio(NL):
    I, _ = fm.spectrum_fine([1, 0, 0, 0], T_true, ne, NL, 0.25, 0.02)
    def integ(w0):
        m = (fm.fine > w0 - 1.5) & (fm.fine < w0 + 1.5)
        return I[m].sum() * fm.step
    return integ(324.754) / integ(327.396)
thin = doublet_ratio(1e11)
# thin-limit integrated energy intensity ~ A*gk*exp(-Ek/kT)/lambda  (database values, incl. different upper levels)
i1 = cu[np.argmin(abs(at.wl[cu] - 324.754))]; i2 = cu[np.argmin(abs(at.wl[cu] - 327.396))]
expect = (at.aki[i1] * at.gk[i1] / at.wl[i1] * np.exp(-at.ek[i1] / (KB_EV * T_true))) / \
         (at.aki[i2] * at.gk[i2] / at.wl[i2] * np.exp(-at.ek[i2] / (KB_EV * T_true)))
check("Cu doublet thin ratio matches A*gk*exp(-Ek/kT)/lambda", abs(thin / expect - 1) < 0.01,
      f"model={thin:.3f} expected={expect:.3f}")
r = [doublet_ratio(NL) for NL in (1e13, 1e15, 1e16, 1e17)]
check("self-absorption drives doublet ratio down monotonically", all(np.diff(r) < 0) and r[-1] < 1.4,
      "ratios at NL=1e13,1e15,1e16,1e17: " + ", ".join(f"{v:.2f}" for v in r))

# ---- Test 3: Saha behaviour ----
f_lowT, _ = fm.ionisation_fractions(6000.0, 1e17)
f_hiT, _ = fm.ionisation_fractions(13000.0, 1e17)
f_hine, _ = fm.ionisation_fractions(9000.0, 1e18)
f_lone, _ = fm.ionisation_fractions(9000.0, 1e16)
check("ionised fraction increases with T", f_hiT[1] > f_lowT[1], f"Cu II frac {f_lowT[1]:.3f} -> {f_hiT[1]:.3f}")
check("ionised fraction decreases with ne", f_hine[1] < f_lone[1], f"Cu II frac ne=1e18: {f_hine[1]:.3f}, ne=1e16: {f_lone[1]:.3f}")
check("fractions sum to 1 per element", all(abs(f_lowT[2 * i] + f_lowT[2 * i + 1] - 1) < 1e-12 for i in range(len(EL))))

# ---- Test 4: thin-limit intensity is proportional to the number fraction (Zn/Pb mixture, no Fe/Cu) ----
wa, wb = [0, 0.5, 0.5, 0], [0, 0.25, 0.75, 0]
Ia, _ = fm.thin_spectrum_fine(wa, T_true, ne, 1e11, 0.4, 0.1)
Ib, _ = fm.thin_spectrum_fine(wb, T_true, ne, 1e11, 0.4, 0.1)
m = (fm.fine > 480.3) & (fm.fine < 481.8)   # Zn I 481.05
meas = Ib[m].sum() / Ia[m].sum()
exp_ratio = fm.number_fractions(wb)[1] / fm.number_fractions(wa)[1]
check("thin Zn 481 intensity proportional to Zn number fraction", abs(meas / exp_ratio - 1) < 0.01,
      f"measured={meas:.4f} expected={exp_ratio:.4f}")

# ---- Timing ----
t0 = time.time()
for _ in range(200):
    I, _ = fm.spectrum_fine([0.6, 0.38, 0.01, 0.01], 8500.0, 5e16, 3e16, 0.5, 0.2)
    fm.to_pixels(I)
dt = (time.time() - t0) / 200
print(f"time per synthetic spectrum: {dt*1e3:.2f} ms")
print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
sys.exit(0 if ok else 1)
