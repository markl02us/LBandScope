"""BPSK modem, framing, and a streaming frame detector/decoder.

The framing here (preamble + scrambler + SYNC/length/CRC) is a self-contained
test waveform used to validate the receiver chain independently of any live
signal. The demodulator stages (frequency estimation, matched filtering, timing
recovery, coherent detection) are the reusable parts.
"""
from __future__ import annotations

import numpy as np


# --- CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) -------------------------
def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc & 0xFFFF


# --- bit/byte packing (MSB first) ------------------------------------------
def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    n = (len(bits) // 8) * 8
    return np.packbits(bits[:n]).tobytes()


# --- additive scrambler: 7-bit LFSR, taps x^7+x^4+1; XOR is self-inverse ----
def pn_sequence(length: int, seed: int = 0x7F) -> np.ndarray:
    state = seed & 0x7F
    out = np.empty(length, dtype=np.uint8)
    for i in range(length):
        bit = ((state >> 6) ^ (state >> 3)) & 1
        out[i] = bit
        state = ((state << 1) | bit) & 0x7F
    return out


def scramble(bits: np.ndarray, seed: int = 0x7F) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint8) ^ pn_sequence(len(bits), seed)).astype(np.uint8)


# --- frame: [SYNC 0x1ACF][LEN u8][PAYLOAD][CRC16]; scrambled after preamble --
SYNC = bytes([0x1A, 0xCF])


def build_frame(payload: bytes) -> bytes:
    if len(payload) > 255:
        raise ValueError("payload too long for u8 length field")
    body = SYNC + bytes([len(payload)]) + payload
    crc = crc16_ccitt(body)
    return body + bytes([crc >> 8, crc & 0xFF])


def parse_frame(data: bytes):
    """Locate SYNC, validate length and CRC. Returns (payload, ok)."""
    idx = data.find(SYNC)
    if idx < 0:
        return None, False
    p = idx + len(SYNC)
    if p >= len(data):
        return None, False
    length = data[p]
    p += 1
    end = p + length
    if end + 2 > len(data):
        return None, False
    payload = data[p:end]
    got = (data[end] << 8) | data[end + 1]
    want = crc16_ccitt(SYNC + bytes([length]) + payload)
    return payload, (got == want)


# --- BPSK: bit 0 -> +1, bit 1 -> -1; rectangular pulse, sps samples/symbol ---
# 64-bit PN preamble sent in the clear (autocorrelation used for sync/phase).
PREAMBLE_BITS = np.array(
    [int(x) for x in
     "1110010010101110110000011101101010010001011111100110001000010011"],
    dtype=np.uint8,
)


def modulate(payload: bytes, sps: int = 8, seed: int = 0x7F) -> np.ndarray:
    frame_bits = scramble(bytes_to_bits(build_frame(payload)), seed)
    all_bits = np.concatenate([PREAMBLE_BITS, frame_bits])
    symbols = 1 - 2 * all_bits.astype(np.float64)
    return np.repeat(symbols, sps).astype(np.complex128)


def apply_channel(iq: np.ndarray, snr_db: float = 12.0, cfo: float = 5e-4,
                  phase: float = 0.7, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply carrier frequency offset, a static phase, and complex AWGN."""
    rng = rng or np.random.default_rng(0)
    n = np.arange(len(iq))
    y = iq * np.exp(1j * (2 * np.pi * cfo * n + phase))
    noise_p = np.mean(np.abs(iq) ** 2) / (10 ** (snr_db / 10))
    noise = rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))
    return y + noise * np.sqrt(noise_p / 2)


# Longest possible frame in symbols: preamble + (SYNC + len + 255B + CRC) * 8.
MAX_FRAME_SYMS = len(PREAMBLE_BITS) + (len(SYNC) + 1 + 255 + 2) * 8


def _coarse_cfo(iq: np.ndarray, sps: int = 8) -> float:
    """Squaring-loop frequency estimate: x^2 collapses BPSK data and leaves a
    tone at 2*cfo, located by FFT + parabolic interpolation. The search is
    band-limited to |2*cfo| < 0.5/sps because a pulse-shaped BPSK, once squared,
    also produces lines at the symbol rate (1/sps) that would otherwise dominate.
    """
    sq = iq ** 2
    sq = sq - np.mean(sq)
    nfft = 4 * len(sq)
    spec = np.abs(np.fft.fft(sq, nfft))
    freqs = np.fft.fftfreq(nfft)
    band = np.abs(freqs) <= 0.5 / sps
    k = int(np.argmax(np.where(band, spec, 0.0)))
    if spec[k] < 6.0 * np.median(spec):           # no prominent tone => cfo ~ 0
        return 0.0
    a, b, c = spec[(k - 1) % nfft], spec[k], spec[(k + 1) % nfft]
    denom = a - 2 * b + c
    delta = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5)) if denom != 0 else 0.0
    f = (k + delta) / nfft
    if f >= 0.5:
        f -= 1.0
    return f / 2.0


def _fft_correlate(r: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Cross-correlation via FFT, equivalent to np.correlate(r, t, 'valid') but
    O(N log N) so it scales to multi-megasample buffers."""
    L, T = len(r), len(template)
    if L < T:
        return np.zeros(0, dtype=np.complex128)
    nfft = 1 << int(np.ceil(np.log2(L + T - 1)))
    C = np.fft.ifft(np.fft.fft(r, nfft) * np.conj(np.fft.fft(template, nfft)))
    return C[:L - T + 1]


def _detect(r: np.ndarray, template: np.ndarray):
    """Return (normalized, magnitude) correlations.

    normalized = |<r,t>| / (||r_window|| * ||t||) in [0,1] is level-independent,
    so a single threshold holds across SDR gain settings; it drives detection.
    magnitude preserves the true peak shape for sub-sample timing. Window energy
    is an O(N) prefix-sum, not an O(N*T) convolution.
    """
    c = _fft_correlate(r, template)
    if len(c) == 0:
        return c.real, c.real
    T = len(template)
    mag = np.abs(c)
    cs = np.concatenate([[0.0], np.cumsum(np.abs(r) ** 2)])
    e = cs[T:] - cs[:-T]
    return mag / (np.sqrt(e * T) + 1e-12), mag


def _frame_symbols(r: np.ndarray, start: float, sps: int, pre_sym: np.ndarray):
    """Recover carrier- and timing-corrected symbols for one frame at fractional
    sample `start`. Returns the complex symbol array (constellation) or None."""
    avail = int((len(r) - np.ceil(start)) // sps)
    if avail <= len(pre_sym):
        return None
    nsyms = min(avail, MAX_FRAME_SYMS)
    pos = start + np.arange(nsyms * sps)
    xp = np.arange(len(r))
    grid = np.interp(pos, xp, r.real) + 1j * np.interp(pos, xp, r.imag)
    syms = grid.reshape(nsyms, sps).mean(axis=1)          # integrate-and-dump

    # Preamble-aided residual frequency then phase. Each half of the preamble is
    # integrated coherently before differencing, giving a low-variance estimate.
    k = min(len(pre_sym), nsyms)
    prod = syms[:k] * pre_sym[:k]
    idx = np.arange(nsyms)
    if k >= 4:
        h = k // 2
        w = np.angle(np.sum(prod[h:2 * h]) * np.conj(np.sum(prod[:h]))) / h
        syms = syms * np.exp(-1j * w * idx)
    theta = np.angle(np.sum(syms[:k] * pre_sym[:k]))
    return syms * np.exp(-1j * theta)


def _decode_at(r: np.ndarray, start: float, sps: int, seed: int,
               pre_sym: np.ndarray):
    """Decode one frame at `start`. Returns (payload, symbols_consumed, symbols)
    or None."""
    syms = _frame_symbols(r, start, sps, pre_sym)
    if syms is None:
        return None
    bits = (syms.real < 0).astype(np.uint8)
    frame_bits = scramble(bits[len(PREAMBLE_BITS):], seed)
    payload, ok = parse_frame(bits_to_bytes(frame_bits))
    if not ok:
        return None
    consumed = len(PREAMBLE_BITS) + (len(SYNC) + 1 + len(payload) + 2) * 8
    return payload, consumed, syms


def decode_frames(iq: np.ndarray, sps: int = 8, seed: int = 0x7F,
                  thresh: float = 0.45, max_frames: int = 100000,
                  with_symbols: bool = False):
    """Detect and decode every frame in a buffer.

    Returns a list of {'sample': int, 'payload': bytes}, plus 'symbols' (the
    constellation) when `with_symbols` is set. One coarse CFO estimate is applied
    to the whole buffer; the per-frame stage removes the residual.
    """
    iq = np.asarray(iq, dtype=np.complex128)
    pre_sym = 1 - 2 * PREAMBLE_BITS.astype(np.float64)
    template = np.repeat(pre_sym, sps).astype(np.complex128)
    if len(iq) < len(template):
        return []

    cfo = _coarse_cfo(iq[:16384], sps)
    r = iq * np.exp(-1j * 2 * np.pi * cfo * np.arange(len(iq)))

    nc, mag = _detect(r, template)
    Lc = len(nc)
    if Lc == 0:
        return []

    # Candidate peaks: above threshold and a local maximum, found vectorized.
    above = nc >= thresh
    ge_left = np.ones(Lc, dtype=bool)
    ge_left[1:] = nc[1:] >= nc[:-1]
    ge_right = np.ones(Lc, dtype=bool)
    ge_right[:-1] = nc[:-1] >= nc[1:]
    cand = np.where(above & ge_left & ge_right)[0]

    frames, last_end = [], -1
    for i in cand:
        if i < last_end or len(frames) >= max_frames:
            continue
        if 0 < i < Lc - 1:
            a, b, c = mag[i - 1], mag[i], mag[i + 1]
            den = a - 2 * b + c
            mu = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den else 0.0
        else:
            mu = 0.0
        res = _decode_at(r, i + mu, sps, seed, pre_sym)
        if res is not None:
            payload, consumed, syms = res
            entry = {"sample": int(i), "payload": payload}
            if with_symbols:
                entry["symbols"] = syms
            frames.append(entry)
            last_end = i + consumed * sps
    return frames


def demodulate(iq: np.ndarray, sps: int = 8, seed: int = 0x7F):
    """Decode the first frame in `iq`. Returns (payload, ok)."""
    fr = decode_frames(iq, sps, seed, max_frames=1)
    if fr:
        return fr[0]["payload"], True
    return None, False


def constellation(iq: np.ndarray, sps: int = 8, max_points: int = 800) -> np.ndarray:
    """Recovered symbol points from the first detected frame (for display)."""
    fr = decode_frames(iq, sps, max_frames=1, with_symbols=True)
    if not fr or "symbols" not in fr[0]:
        return np.zeros(0, dtype=np.complex128)
    s = fr[0]["symbols"]
    if len(s) > max_points:
        s = s[np.linspace(0, len(s) - 1, max_points).astype(int)]
    return s
