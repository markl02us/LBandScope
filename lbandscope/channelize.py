"""Digital down-conversion and filter design.

`ddc` translates a channel to baseband, low-pass filters it, and resamples to a
requested rate, so a decoder written for a fixed samples-per-symbol can run on
IQ captured at any device rate.
"""
from __future__ import annotations

import numpy as np


def design_lowpass(cutoff: float, num_taps: int = 127) -> np.ndarray:
    """Windowed-sinc low-pass FIR. `cutoff` is in cycles/sample (0..0.5)."""
    if num_taps % 2 == 0:
        num_taps += 1
    n = np.arange(num_taps) - (num_taps - 1) / 2.0
    h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hanning(num_taps)
    return h / np.sum(h)


def rrc(beta: float, sps: int, span: int = 8) -> np.ndarray:
    """Root-raised-cosine FIR, unit energy, spanning `span` symbols each side."""
    N = span * sps
    t = (np.arange(N + 1) - N / 2.0) / sps
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = (np.sin(np.pi * ti * (1 - beta))
                   + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta)))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


def matched_filter(iq: np.ndarray, taps: np.ndarray):
    """Convolve with `taps`; return (filtered, group_delay_in_samples)."""
    delay = (len(taps) - 1) / 2.0
    return np.convolve(iq, taps, mode="full")[:len(iq) + len(taps) - 1], delay


def ddc(iq: np.ndarray, fs: float, f_offset: float, out_rate: float) -> np.ndarray:
    """Mix `f_offset` to DC, anti-alias, and resample from `fs` to `out_rate`.

    Rates share the same unit as `fs` (Hz, or normalized). Resampling is
    fractional, so `out_rate` is met exactly and the caller gets an integer
    samples-per-symbol independent of the device rate.
    """
    iq = np.asarray(iq, dtype=np.complex128)
    n = np.arange(len(iq))
    mixed = iq * np.exp(-1j * 2 * np.pi * (f_offset / fs) * n)

    cutoff = 0.5 * (out_rate / fs) * 0.9
    if cutoff < 0.5:
        mixed = np.convolve(mixed, design_lowpass(cutoff, num_taps=127), mode="same")

    ratio = out_rate / fs
    m = int(np.floor(len(mixed) * ratio))
    if m <= 1:
        return np.zeros(0, dtype=np.complex128)
    pos = np.arange(m) / ratio
    xp = np.arange(len(mixed))
    return (np.interp(pos, xp, mixed.real)
            + 1j * np.interp(pos, xp, mixed.imag)).astype(np.complex128)
