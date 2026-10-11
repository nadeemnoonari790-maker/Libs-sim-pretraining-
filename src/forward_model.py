"""LTE forward model for laser-induced plasma emission (homogeneous slab).

Physics
-------
* Species populations: Saha ionisation balance (neutral / singly ionised) at electron density ne,
  Boltzmann level populations, partition functions summed directly from tabulated energy levels
  (levels above the ionisation potential are discarded).
* Radiative transfer through a uniform slab at temperature T:
        I(lambda) = B_lambda(T) * (1 - exp(-tau(lambda)))
  which reduces to the optically thin limit for small tau and includes self-absorption.
* tau(lambda) = sum_i  A_i * Wlin_i * V(lambda - lambda_i)  where the integrated line optical depth is
        int tau d(lambda) = lambda0^4 /(8 pi c) * A_ki * g_k * (n_i L / g_i) * (1 - exp(-h nu / kT))
  with n_i L = NL * x_el * f_ion * g_i/U(T) * exp(-E_i/kT)   (NL = heavy-particle column density, cm^-2).
* Line shape: Voigt (Gaussian instrument + Lorentzian Stark/pressure part), identical for all lines
  within one spectrum (simplification), evaluated by FFT convolution on a fine wavelength grid.

Atomic data: NIST ASD-derived line database (see data/atomic_lines.db; checked against NIST values).
"""
import sqlite3
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import voigt_profile

H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23
KB_EV = 8.617333262e-5
EV = 1.602176634e-19
SAHA_PREF = 4.829e15  # 2*(2 pi m_e k/h^2)^{3/2}  [cm^-3 K^-3/2]


class AtomicData:
    def __init__(self, db_path, elements, wl_lo, wl_hi, tgrid=None, min_agk=1e5):
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.elements = list(elements)
        self.tgrid = np.arange(3000.0, 20001.0, 100.0) if tgrid is None else tgrid
        self.mass = {}
        self.ip = {}
        sp_rows = {(r[0], r[1]): (r[2], r[3]) for r in
                   con.execute("select element,sp_num,ip_ev,atomic_mass from species_physics")}
        self.species = []  # (element index, stage)
        self.U = []        # partition function tables
        for ie, el in enumerate(self.elements):
            self.mass[el] = sp_rows[(el, 1)][1]
            for stage in (1, 2):
                if (el, stage) not in sp_rows:
                    raise KeyError((el, stage))
                ip = sp_rows[(el, stage)][0]
                self.ip[(el, stage)] = ip
                lv = np.array(con.execute(
                    "select g_level, energy_ev from energy_levels where element=? and sp_num=? and energy_ev<?",
                    (el, stage, ip)).fetchall(), float)
                if len(lv) == 0:
                    lv = np.array([[1.0, 0.0]])
                self.species.append((ie, stage))
                self.U.append((lv[:, 0][None, :] *
                               np.exp(-lv[:, 1][None, :] / (KB_EV * self.tgrid[:, None]))).sum(axis=1))
        self.U = np.stack(self.U)  # (n_species, n_T)
        sp_index = {s: k for k, s in enumerate(self.species)}
        cols = ("element", "sp_num", "wavelength_nm", "aki", "ei_ev", "ek_ev", "gi", "gk")
        rows = []
        for ie, el in enumerate(self.elements):
            for stage in (1, 2):
                ip = self.ip[(el, stage)]
                q = con.execute(
                    "select wavelength_nm,aki,ei_ev,ek_ev,gi,gk from lines where element=? and sp_num=? "
                    "and wavelength_nm between ? and ? and aki>0 and gi>0 and gk>0 and ek_ev<? and aki*gk>=?",
                    (el, stage, wl_lo, wl_hi, ip, min_agk)).fetchall()
                for r in q:
                    rows.append((sp_index[(ie, stage)],) + tuple(r))
        a = np.array(rows, float)
        self.sp_idx = a[:, 0].astype(int)
        self.wl, self.aki, self.ei, self.ek, self.gi, self.gk = (a[:, k] for k in range(1, 7))
        self.nu_ev = 1239.84198 / self.wl  # photon energy [eV]
        self.n_lines = len(self.wl)
        self.ion_pot = np.array([self.ip[(self.elements[ie], st)] for ie, st in self.species])
        self.is_neutral = np.array([st == 1 for ie, st in self.species])
        self.elem_of_sp = np.array([ie for ie, st in self.species])
        self.masses = np.array([self.mass[e] for e in self.elements])

    def Ut(self, T):
        return np.array([np.interp(T, self.tgrid, u) for u in self.U])


class LIBSForward:
    def __init__(self, atomic, wl_pix, fine_step=0.05, pad=8.0, kernel_half=12.0):
        self.at = atomic
        self.wl_pix = np.asarray(wl_pix, float)
        self.step = fine_step
        self.fine = np.arange(self.wl_pix[0] - pad, self.wl_pix[-1] + pad, fine_step)
        self.kernel_x = np.arange(-kernel_half, kernel_half + fine_step / 2, fine_step)
        self.bin = np.clip(np.round((self.at.wl - self.fine[0]) / fine_step).astype(int), 0, len(self.fine) - 1)
        lam_cm = self.at.wl * 1e-7
        # constant part of integrated line optical depth [cm] per (A*gk*n_i L/g_i...), converted to nm below
        self.k0 = (lam_cm ** 4 / (8 * np.pi * (C * 100.0))) * 1e7  # nm per (s^-1 cm^-2)
        # hydrogenic-like Stark scaling: width ~ (ne/1e17) * (2 eV/(IP-Ek))^2 * (lambda/500nm)^2   [relative units]
        ipd = np.maximum(self.at.ion_pot[self.at.sp_idx] - self.at.ek, 0.3)
        self.ipd = ipd
        self.stark_scale = (2.0 / ipd) ** 2 * (self.at.wl / 500.0) ** 2
        self.n_groups = 10
        self.group_edges = np.geomspace(0.02, 8.0, self.n_groups + 1)
        self.group_centres = np.sqrt(self.group_edges[:-1] * self.group_edges[1:])

    # ---------- plasma state -------------
    def number_fractions(self, mass_fractions):
        w = np.asarray(mass_fractions, float)
        x = w / self.at.masses
        return x / x.sum()

    def ionisation_fractions(self, T, ne):
        """Return fraction of each element's atoms in each species (neutral / singly ionised)."""
        at = self.at
        U = at.Ut(T)
        f = np.zeros(len(at.species))
        for ie in range(len(at.elements)):
            k1, k2 = 2 * ie, 2 * ie + 1
            S = SAHA_PREF * T ** 1.5 * (U[k2] / U[k1]) * np.exp(-at.ion_pot[k1] / (KB_EV * T))
            r = S / ne
            f[k1] = 1.0 / (1.0 + r)
            f[k2] = r / (1.0 + r)
        return f, U

    # ---------- spectrum -------------
    def tau_comb(self, mass_fractions, T, ne, NL):
        at = self.at
        x = self.number_fractions(mass_fractions)
        f, U = self.ionisation_fractions(T, ne)
        n_sp = NL * x[at.elem_of_sp] * f  # column density per species [cm^-2]
        pop = n_sp[at.sp_idx] / U[at.sp_idx] * np.exp(-at.ei / (KB_EV * T))  # n_i L / g_i
        stim = 1.0 - np.exp(-at.nu_ev / (KB_EV * T))
        area = self.k0 * at.aki * at.gk * pop * stim  # integrated optical depth [nm]
        comb = np.bincount(self.bin, weights=area, minlength=len(self.fine)) / self.step
        return comb

    def planck(self, T):
        lam = self.fine * 1e-9
        x = H * C / (lam * KB * T)
        return 2 * H * C ** 2 / lam ** 5 / np.expm1(x)

    def spectrum_fine(self, mass_fractions, T, ne, NL, fwhm_g, fwhm_l):
        comb = self.tau_comb(mass_fractions, T, ne, NL)
        sig = fwhm_g / 2.3548200450309493
        gam = fwhm_l / 2.0
        ker = voigt_profile(self.kernel_x, sig, gam) * self.step
        tau = fftconvolve(comb, ker, mode="same")
        B = self.planck(T)
        return B * (-np.expm1(-tau)), tau

    def _tau_stark(self, mass_fractions, T, ne, NL, fwhm_g, stark_ref, stark_p, min_l=0.02):
        comb_all = self.tau_comb_lines(mass_fractions, T, ne, NL)          # per-line integrated tau [nm]
        scale = (2.0 / self.ipd) ** stark_p * (self.at.wl / 500.0) ** 2
        w = np.clip(stark_ref * (ne / 1e17) * scale, min_l, 8.0)
        g = np.clip(np.digitize(w, self.group_edges) - 1, 0, self.n_groups - 1)
        nf = len(self.fine)
        combs = np.bincount(g * nf + self.bin, weights=comb_all, minlength=self.n_groups * nf).reshape(self.n_groups, nf) / self.step
        sig = fwhm_g / 2.3548200450309493
        tau = np.zeros(nf)
        for k in range(self.n_groups):
            if combs[k].any():
                ker = voigt_profile(self.kernel_x, sig, self.group_centres[k] / 2.0) * self.step
                tau += fftconvolve(combs[k], ker, mode="same")
        return tau

    def spectrum_fine_stark(self, mass_fractions, T, ne, NL, fwhm_g, stark_ref, stark_p=2.0, layer2=None, thin=False):
        """Slab spectrum with per-line Lorentzian (Stark) widths  w_i = stark_ref*(ne/1e17)*(2eV/(IP-Ek))^p*(lam/500)^2 [nm FWHM].
        layer2=(T2, ne2, NL2) adds a cooler absorbing layer in front of the hot core (self-reversal):
            I = B(T1)(1-e^-tau1) e^-tau2 + B(T2)(1-e^-tau2)."""
        tau = self._tau_stark(mass_fractions, T, ne, NL, fwhm_g, stark_ref, stark_p)
        B = self.planck(T)
        I = B * tau if thin else B * (-np.expm1(-tau))
        if layer2 is not None:
            T2, ne2, NL2 = layer2
            tau2 = self._tau_stark(mass_fractions, T2, ne2, NL2, fwhm_g, stark_ref, stark_p)
            I = I * np.exp(-tau2) + self.planck(T2) * (-np.expm1(-tau2))
            tau = tau + tau2
        return I, tau

    def tau_comb_lines(self, mass_fractions, T, ne, NL):
        at = self.at
        x = self.number_fractions(mass_fractions)
        f, U = self.ionisation_fractions(T, ne)
        n_sp = NL * x[at.elem_of_sp] * f
        pop = n_sp[at.sp_idx] / U[at.sp_idx] * np.exp(-at.ei / (KB_EV * T))
        stim = 1.0 - np.exp(-at.nu_ev / (KB_EV * T))
        return self.k0 * at.aki * at.gk * pop * stim

    def thin_spectrum_fine(self, mass_fractions, T, ne, NL, fwhm_g, fwhm_l):
        """Optically thin limit (for tests): I = B * tau."""
        comb = self.tau_comb(mass_fractions, T, ne, NL)
        sig = fwhm_g / 2.3548200450309493
        ker = voigt_profile(self.kernel_x, sig, fwhm_l / 2.0) * self.step
        tau = fftconvolve(comb, ker, mode="same")
        return self.planck(T) * tau, tau

    def to_pixels(self, I_fine, shift=0.0, stretch=0.0):
        wl = self.wl_pix * (1.0 + stretch) + shift
        return np.interp(wl, self.fine, I_fine)
