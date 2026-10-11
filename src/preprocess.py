"""Preprocessing applied IDENTICALLY to real and synthetic spectra.

1. Continuum / camera-offset removal: morphological opening (running minimum over `win` pixels,
   then running maximum) followed by Gaussian smoothing gives a lower envelope; it is subtracted.
2. Negative values are clipped to zero.
3. Each spectrum is scaled to unit L2 norm (compositional, shot-intensity independent).
"""
import numpy as np
from scipy.ndimage import minimum_filter1d, maximum_filter1d, gaussian_filter1d

WIN = 61      # pixels (~14 nm at 0.233 nm/px) - wider than any isolated line
SMOOTH = 15.0


def baseline(X, win=WIN, smooth=SMOOTH):
    X = np.atleast_2d(X).astype(np.float64)
    b = minimum_filter1d(X, size=win, axis=1, mode="nearest")
    b = maximum_filter1d(b, size=win, axis=1, mode="nearest")
    return gaussian_filter1d(b, sigma=smooth, axis=1, mode="nearest")


def preprocess(X, win=WIN, smooth=SMOOTH, return_baseline=False):
    X = np.atleast_2d(X).astype(np.float64)
    b = baseline(X, win, smooth)
    Y = np.clip(X - b, 0.0, None)
    n = np.linalg.norm(Y, axis=1, keepdims=True)
    Y = Y / np.maximum(n, 1e-12)
    return (Y.astype(np.float32), b) if return_baseline else Y.astype(np.float32)
