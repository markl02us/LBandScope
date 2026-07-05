"""Spectrum and waterfall computation.

Pure NumPy: produces power spectra and colorized image rows. Rendering to the
screen is left to the UI so this stays testable and dependency-free.
"""
from __future__ import annotations

import numpy as np


def spectrum_db(iq: np.ndarray, nfft: int = 512) -> np.ndarray:
    """Averaged periodogram of a block, in dB, DC-centered (fftshift)."""
    x = np.asarray(iq, dtype=np.complex128)
    if len(x) < nfft:
        x = np.concatenate([x, np.zeros(nfft - len(x), dtype=np.complex128)])
    nseg = len(x) // nfft
    seg = x[:nseg * nfft].reshape(nseg, nfft)
    win = np.hanning(nfft)
    power = np.abs(np.fft.fft(seg * win, axis=1)) ** 2
    p = np.fft.fftshift(power.mean(axis=0))
    return 10.0 * np.log10(p + 1e-12)


# Perceptually ordered colormap control points (dark -> bright), like viridis.
_CMAP = np.array([
    (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
    (38, 130, 142), (31, 158, 137), (53, 183, 121), (110, 206, 88),
    (181, 222, 43), (253, 231, 37),
], dtype=np.float64)


def colorize(rows_db: np.ndarray, lo: float | None = None,
             hi: float | None = None) -> np.ndarray:
    """Map dB values to RGB uint8 using the colormap. Accepts a 1-D row or a
    2-D stack of rows; returns the same shape with a trailing RGB axis."""
    a = np.asarray(rows_db, dtype=np.float64)
    lo = float(np.percentile(a, 5)) if lo is None else lo
    hi = float(np.percentile(a, 99)) if hi is None else hi
    if hi <= lo:
        hi = lo + 1.0
    t = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    idx = t * (len(_CMAP) - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, len(_CMAP) - 1)
    frac = (idx - i0)[..., None]
    rgb = _CMAP[i0] * (1 - frac) + _CMAP[i1] * frac
    return rgb.astype(np.uint8)


def to_ppm(rgb: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 array as binary PPM (P6)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    return f"P6 {w} {h} 255\n".encode() + rgb.tobytes()
