"""Front-end signal conditioning and quality metrics.

These are the corrections a receiver applies before demodulation to clean up a
low-cost SDR front end, plus simple metrics used to drive a signal meter so a
user can aim an antenna for best reception.
"""
from __future__ import annotations

import numpy as np


def remove_dc(iq: np.ndarray) -> np.ndarray:
    """Remove the DC component. Cheap RTL-SDR front ends place a spike at the
    tuned frequency; subtracting the mean suppresses it."""
    iq = np.asarray(iq)
    return iq - iq.mean()


def correct_iq_imbalance(iq: np.ndarray) -> np.ndarray:
    """Blind correction of I/Q gain and phase imbalance. Orthogonalizes Q with
    respect to I and equalizes their power, which reduces the mirror-image
    response that otherwise leaks energy across the spectrum."""
    iq = np.asarray(iq)
    i = iq.real.astype(np.float64)
    q = iq.imag.astype(np.float64)
    i -= i.mean()
    q -= q.mean()
    ii = np.mean(i * i)
    if ii <= 0:
        return iq
    q -= (np.mean(i * q) / ii) * i          # phase: decorrelate Q from I
    qq = np.mean(q * q)
    if qq > 0:
        q *= np.sqrt(ii / qq)               # gain: equalize amplitude
    return (i + 1j * q).astype(iq.dtype if np.iscomplexobj(iq) else np.complex128)


def precondition(iq: np.ndarray) -> np.ndarray:
    """Apply the standard conditioning chain."""
    return correct_iq_imbalance(remove_dc(iq))


def signal_level_db(iq: np.ndarray) -> float:
    """Average power in dB relative to full scale (level meter)."""
    p = float(np.mean(np.abs(np.asarray(iq)) ** 2))
    return 10.0 * np.log10(p + 1e-12)


def quality_db(spectrum_row_db: np.ndarray) -> float:
    """Peak-to-noise-floor prominence in dB from a power spectrum, a robust
    proxy for how strong the signal is above the noise."""
    s = np.asarray(spectrum_row_db)
    return float(np.max(s) - np.median(s))
