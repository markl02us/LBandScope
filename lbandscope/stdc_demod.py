"""STD-C symbol demodulator: coherent BPSK recovery from baseband IQ.

This follows the proven Scytale-C lineage (coherent BPSK, with the 180-degree
polarity ambiguity resolved at the unique word) rather than a per-symbol
differential detector. That keeps it consistent with the decoder in `stdc.py`,
whose frame sync searches both polarities.

Chain:
  coarse carrier estimate (squaring) -> derotate
  fractional resample to an integer samples-per-symbol
  root-raised-cosine matched filter
  timing: pick the maximum-eye-opening sample phase (parabolic sub-sample refine)
  residual carrier: squaring estimate of frequency then phase -> derotate
  slice the real part -> 0/1 symbols

`demodulate_symbols` returns the raw symbol stream; `receive` runs frame sync and
the full decode, returning decoded frames. Validated end to end against a
synthetic pulse-shaped, carrier-offset, noisy channel (see `_selftest`).
"""
from __future__ import annotations

import numpy as np

from . import channelize, stdc

SYMBOL_RATE = 1200.0


def _remove_coarse_cfo(x: np.ndarray, fs: float, symbol_rate: float) -> np.ndarray:
    """Squaring carrier estimate: x**2 removes BPSK data and leaves a tone at
    twice the carrier offset. The tone is found by FFT and refined to sub-bin by
    parabolic interpolation for a precise, low-variance estimate. The search
    spans +/-0.8*symbol_rate (so carrier offsets up to +/-0.4*symbol_rate are
    acquired) while staying clear of the +/-symbol_rate pulse-shaping lines."""
    sq = x * x
    sq = sq - sq.mean()
    n = len(sq)
    spec = np.abs(np.fft.fft(sq))
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    band = np.abs(freqs) < 0.8 * symbol_rate
    k = int(np.argmax(np.where(band, spec, 0.0)))
    if spec[k] < 6.0 * np.median(spec):
        return x
    a, b, c = spec[(k - 1) % n], spec[k], spec[(k + 1) % n]
    den = a - 2 * b + c
    delta = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den != 0 else 0.0
    cfo = (freqs[k] + delta * fs / n) / 2.0
    return x * np.exp(-1j * 2 * np.pi * (cfo / fs) * np.arange(n))


def _resample_to_sps(x: np.ndarray, fs: float, symbol_rate: float, sps: int) -> np.ndarray:
    target = symbol_rate * sps
    if abs(fs - target) < 1e-6:
        return x
    m = int(np.floor(len(x) * target / fs))
    pos = np.arange(m) * (fs / target)
    xp = np.arange(len(x))
    return np.interp(pos, xp, x.real) + 1j * np.interp(pos, xp, x.imag)


def _symbols(iq: np.ndarray, samp_rate: float, symbol_rate: float = SYMBOL_RATE,
             beta: float = 0.3, sps: int = 8) -> np.ndarray:
    """Baseband IQ -> complex symbol stream (carrier not yet removed). Coarse
    carrier estimate, resample to integer sps, matched filter, eye-opening
    timing."""
    x = np.asarray(iq, dtype=np.complex128)
    if len(x) < sps * 4:
        return np.zeros(0, dtype=np.complex128)
    x = _remove_coarse_cfo(x, samp_rate, symbol_rate)
    x = _resample_to_sps(x, samp_rate, symbol_rate, sps)
    x = np.convolve(x, channelize.rrc(beta, sps), mode="same")
    usable = (len(x) // sps) * sps
    if usable < sps:
        return np.zeros(0, dtype=np.complex128)
    grid = x[:usable].reshape(-1, sps)
    phase = int(np.argmax(np.mean(np.abs(grid) ** 2, axis=0)))
    return grid[:, phase]


def _derotate(syms: np.ndarray) -> np.ndarray:
    """Remove the carrier phase (squaring estimate) from a block of symbols.
    Returns the corrected complex symbols; the real part is the soft bit value
    and its sign is the hard bit. Global polarity is left for the unique word.

    Frequency is handled up front by the coarse estimate in `_symbols`, which
    runs over the whole capture with fine resolution; a per-frame frequency term
    is not only unnecessary but harmful, because on a short block the squared
    spectrum's data self-noise can outweigh the small residual carrier tone."""
    syms = np.asarray(syms, dtype=np.complex128)
    if len(syms) > 1:
        theta = np.angle(np.sum(syms * syms)) / 2.0
        syms = syms * np.exp(-1j * theta)
    return syms


def demodulate_symbols(iq: np.ndarray, samp_rate: float,
                       symbol_rate: float = SYMBOL_RATE, beta: float = 0.3,
                       sps: int = 8) -> np.ndarray:
    """Baseband IQ -> hard symbol stream (0/1), carrier removed over the whole
    buffer. Suitable for a single frame; `receive` corrects per frame."""
    return (_derotate(_symbols(iq, samp_rate, symbol_rate, beta, sps)).real < 0).astype(np.uint8)


def receive(iq: np.ndarray, samp_rate: float, symbol_rate: float = SYMBOL_RATE,
            tolerance: int = 30, soft: bool = True, **kw) -> list[dict]:
    """Full STD-C receive: locate every frame and decode it (soft-decision by
    default) with carrier corrected per frame, so a drifting or multi-frame
    capture decodes. Returns decoded frames.

    The coarse carrier estimate removes the frequency offset over the whole
    capture, so one pass of the unique-word finder locates every frame; each is
    then phase-corrected and decoded independently."""
    syms = _symbols(iq, samp_rate, symbol_rate, **kw)
    frame = stdc.FRAME_SYMBOLS
    if len(syms) < frame:
        return []
    provisional = (_derotate(syms).real < 0).astype(np.uint8)
    out = []
    for offset, _rev in stdc.find_uw(provisional, tolerance):
        seg = syms[offset:offset + frame]
        if len(seg) < frame:
            continue
        d = _derotate(seg)                            # per-frame phase correction
        hits = stdc.find_uw((d.real < 0).astype(np.uint8), tolerance)
        if not hits:
            continue
        real = -d.real if hits[0][1] else d.real      # align polarity: + for a 0 bit
        fr = stdc.decode_soft(real) if soft else stdc.decode_frame((real < 0).astype(np.uint8))
        fr["offset"] = int(offset)
        out.append(fr)
    return out


# --- synthetic transmit + channel, for validation --------------------------
def modulate_iq(sym: np.ndarray, sps: int = 8, beta: float = 0.3) -> np.ndarray:
    """Pulse-shaped BPSK: 0 -> +1, 1 -> -1, root-raised-cosine shaped."""
    bpsk = 1.0 - 2.0 * np.asarray(sym, dtype=np.float64)
    up = np.zeros(len(bpsk) * sps)
    up[::sps] = bpsk
    return np.convolve(up, channelize.rrc(beta, sps), mode="full")


def _selftest(snr_db: float = 12.0, cfo_hz: float = 120.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, stdc.INFO_BYTES, dtype=np.uint8)
    sym = stdc.encode_frame(frame.tobytes())
    sps = 8
    fs = SYMBOL_RATE * sps
    tx = modulate_iq(sym, sps)
    n = np.arange(len(tx))
    rx = tx * np.exp(1j * (2 * np.pi * (cfo_hz / fs) * n + 0.7))
    p = np.mean(np.abs(tx) ** 2) / (10 ** (snr_db / 10))
    rx = rx + np.sqrt(p / 2) * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))
    frames = receive(rx, fs)
    ok = bool(frames) and frames[0]["bytes"] == frame.tobytes()
    return {"decoded": len(frames), "byte_exact": ok,
            "frame_number": frames[0]["frame_number"] if frames else None}


if __name__ == "__main__":
    print(_selftest())
