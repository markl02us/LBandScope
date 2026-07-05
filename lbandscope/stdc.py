"""Inmarsat-C (STD-C) TDM receive chain: unique-word framing, row
depermutation, deinterleaving, K=7 rate-1/2 Viterbi FEC, and descrambling —
plus the matching transmit chain used to validate it.

This is an independent implementation written from the public description of the
signal (frame geometry, permutation, interleaver, convolutional code, and
scrambler as documented in the GPL-3.0 Scytale-C lineage). It shares no code with
those projects.

What the round-trip self-test proves: a random 640-byte frame pushed through the
full transmit chain and back returns byte-exact, and the Viterbi stage corrects
a realistic number of symbol errors. That validates the framing, permutation,
interleaver, convolutional code, and scrambler as correct and mutually inverse.

What it does NOT prove on its own: reception of a live off-air signal. That also
requires the symbol demodulator (carrier and timing recovery from IQ) and a real
capture, which are validated separately.

Frame geometry (one TDM frame):
    10368 channel symbols = 64 rows x 162 columns
    each row  = 2 unique-word symbols + 160 data symbols
    64 x 160  = 10240 coded bits --Viterbi--> 5120 bits = 640 bytes
"""
from __future__ import annotations

import numpy as np

# --- frame geometry ---------------------------------------------------------
FRAME_SYMBOLS = 10368
ROWS, COLS = 64, 160
CODED_BITS = 10240
INFO_BYTES = 640

# 64-symbol unique word: one bit per interleaver row, transmitted twice per row.
UW = np.array([
    0, 0, 0, 0,  0, 1, 1, 1,  1, 1, 1, 0,  1, 0, 1, 0,
    1, 1, 0, 0,  1, 1, 0, 1,  1, 1, 0, 1,  1, 0, 1, 0,
    0, 1, 0, 0,  1, 1, 1, 0,  0, 0, 1, 0,  1, 1, 1, 1,
    0, 0, 1, 0,  1, 0, 0, 0,  1, 1, 0, 0,  0, 0, 1, 0,
], dtype=np.uint8)

# --- row permutation (transmitted row order) --------------------------------
_ROW = np.arange(ROWS)
PERM = (_ROW * 23) % ROWS          # depermuted block i <- transmitted block PERM[i]
PERM_INV = np.argsort(PERM)        # transmitted block p carries data/UW of row PERM_INV[p]

# --- interleaver index maps -------------------------------------------------
# Data symbol at transmitted position PERM[i]*162 + 2 + c carries coded bit c*64+i.
_I, _C = np.meshgrid(_ROW, np.arange(COLS), indexing="ij")   # (64,160)
_CODED_POS = (_C * ROWS + _I).ravel()
_TX_POS = (PERM[_I] * 162 + 2 + _C).ravel()
_UW_POS0 = PERM * 162
_UW_POS1 = PERM * 162 + 1

# coded = sym[_DECODE_GATHER]
_DECODE_GATHER = np.empty(CODED_BITS, dtype=np.intp)
_DECODE_GATHER[_CODED_POS] = _TX_POS

# --- convolutional code: K=7, rate 1/2, NASA-standard polynomials ------------
G1, G2 = 0o171, 0o133


def _parity(x: int) -> int:
    p = 0
    while x:
        p ^= x & 1
        x >>= 1
    return p


_PC1 = np.array([_parity(r & G1) for r in range(128)], dtype=np.uint8)
_PC2 = np.array([_parity(r & G2) for r in range(128)], dtype=np.uint8)

# Trellis: for each next state t, its two predecessors and the expected output
# pair on each incoming edge (input bit = t & 1).
_T = np.arange(ROWS)
_PREV0 = _T >> 1
_PREV1 = (_T >> 1) | 32
_BT = _T & 1
_REG0 = (_PREV0 << 1) | _BT
_REG1 = (_PREV1 << 1) | _BT
_EO0A, _EO0B = _PC1[_REG0], _PC2[_REG0]
_EO1A, _EO1B = _PC1[_REG1], _PC2[_REG1]


def conv_encode(bits: np.ndarray) -> np.ndarray:
    """K=7 rate-1/2 convolutional encoder, G1 output first."""
    bits = np.asarray(bits, dtype=np.uint8)
    out = np.empty(bits.size * 2, dtype=np.uint8)
    s = 0
    for i in range(bits.size):
        reg = (s << 1) | int(bits[i])
        out[2 * i] = _PC1[reg]
        out[2 * i + 1] = _PC2[reg]
        s = reg & 0x3F
    return out


def viterbi_decode(coded: np.ndarray) -> np.ndarray:
    """Hard-decision Viterbi for the K=7 rate-1/2 code. Returns the info bits.
    Traceback starts from the best (minimum-metric) end state."""
    coded = np.asarray(coded, dtype=np.uint8)
    n = coded.size // 2
    r = coded[: 2 * n].reshape(n, 2)
    inf = 1 << 30
    pm = np.full(ROWS, inf, dtype=np.int64)
    pm[0] = 0
    dec = np.empty((n, ROWS), dtype=np.uint8)
    for k in range(n):
        r0, r1 = int(r[k, 0]), int(r[k, 1])
        c0 = pm[_PREV0] + (_EO0A ^ r0) + (_EO0B ^ r1)
        c1 = pm[_PREV1] + (_EO1A ^ r0) + (_EO1B ^ r1)
        sel = c1 < c0
        pm = np.where(sel, c1, c0)
        dec[k] = np.where(sel, _PREV1, _PREV0).astype(np.uint8)
    s = int(np.argmin(pm))
    bits = np.empty(n, dtype=np.uint8)
    for k in range(n - 1, -1, -1):
        bits[k] = s & 1
        s = int(dec[k, s])
    return bits


# --- scrambler --------------------------------------------------------------
def _descrambler_array() -> np.ndarray:
    reg = 0x80
    a = np.empty(160, dtype=np.uint8)
    for i in range(160):
        x7 = reg & 1
        a[i] = x7
        x5 = (reg >> 2) & 1
        x4 = (reg >> 3) & 1
        x3 = (reg >> 4) & 1
        nb = x7 ^ x5 ^ x4 ^ x3
        reg = (reg >> 1) | (nb << 7)
    return a


_DESCR = _descrambler_array()
_BITREV = np.array(
    [int(f"{b:08b}"[::-1], 2) for b in range(256)], dtype=np.uint8
)


def scramble_bytes(data: np.ndarray) -> np.ndarray:
    """Per-byte bit reversal plus group complement. Self-inverse: the same
    operation both scrambles (transmit) and descrambles (receive)."""
    out = _BITREV[np.asarray(data, dtype=np.uint8)].copy()
    grp = out.reshape(160, 4)
    grp[_DESCR.astype(bool)] ^= 0xFF
    return out


# --- full frame transmit / receive ------------------------------------------
def encode_frame(plaintext: bytes | np.ndarray) -> np.ndarray:
    """640 information bytes -> 10368 channel symbols (0/1)."""
    data = np.frombuffer(bytes(plaintext), dtype=np.uint8)
    if data.size != INFO_BYTES:
        raise ValueError(f"frame must be {INFO_BYTES} bytes, got {data.size}")
    coded = conv_encode(np.unpackbits(scramble_bytes(data)))
    sym = np.zeros(FRAME_SYMBOLS, dtype=np.uint8)
    sym[_TX_POS] = coded[_CODED_POS]
    sym[_UW_POS0] = UW
    sym[_UW_POS1] = UW
    return sym


def decode_frame(sym: np.ndarray) -> dict:
    """10368 channel symbols (0/1) -> decoded frame.
    Returns {'bytes': 640-byte payload, 'frame_number': int}."""
    sym = np.asarray(sym, dtype=np.uint8)
    if sym.size != FRAME_SYMBOLS:
        raise ValueError(f"frame must be {FRAME_SYMBOLS} symbols, got {sym.size}")
    bits = viterbi_decode(sym[_DECODE_GATHER])
    payload = scramble_bytes(np.packbits(bits))
    return {
        "bytes": payload.tobytes(),
        "frame_number": (int(payload[2]) << 8) | int(payload[3]),
    }


def find_uw(sym: np.ndarray, tolerance: int = 30):
    """Slide a full-frame window over a symbol stream and return the start
    offsets where the unique-word distribution matches (normal or reversed
    polarity). This is the frame-synchronisation step for a continuous stream."""
    sym = np.asarray(sym, dtype=np.uint8)
    if sym.size < FRAME_SYMBOLS:
        return []
    # UW symbols sit at row starts: offsets 0,1, 162,163, ... within a frame.
    # On air, transmit block p carries the UW bit of row PERM_INV[p].
    row = np.arange(ROWS) * 162
    uw_idx = np.concatenate([row, row + 1])
    uw_tx = UW[PERM_INV]
    uw_bits = np.concatenate([uw_tx, uw_tx]).astype(np.int16)
    n = uw_bits.size

    # Error (allowing reversed polarity) at every candidate offset.
    span = sym.size - FRAME_SYMBOLS + 1
    raw = np.empty(span, dtype=np.int16)
    for start in range(span):
        raw[start] = int(np.count_nonzero(sym[start + uw_idx].astype(np.int16) != uw_bits))
    err = np.minimum(raw, n - raw)

    # Because the two UW symbols per row are identical, an offset-by-one half
    # matches; only accept an offset that is a strict local minimum of the error
    # and below tolerance, then keep detections at least one frame apart.
    hits = []
    last = -FRAME_SYMBOLS
    guard = 4
    for start in range(span):
        if err[start] > tolerance:
            continue
        lo, hi = max(0, start - guard), min(span, start + guard + 1)
        if err[start] != err[lo:hi].min():
            continue
        if start - last >= FRAME_SYMBOLS:
            hits.append((start, bool(raw[start] > n - raw[start])))  # (offset, reversed)
            last = start
    return hits


# --- self-test --------------------------------------------------------------
def _selftest(trials: int = 20, seed: int = 0) -> dict:
    """Encode/decode round trip, clean and with injected symbol errors."""
    rng = np.random.default_rng(seed)
    clean_ok = 0
    corrected = 0
    data_pos = _TX_POS  # positions that carry FEC-coded data
    for _ in range(trials):
        frame = rng.integers(0, 256, INFO_BYTES, dtype=np.uint8)
        sym = encode_frame(frame.tobytes())
        if decode_frame(sym)["bytes"] == frame.tobytes():
            clean_ok += 1
        # inject errors into ~1.5% of the data-carrying symbols
        noisy = sym.copy()
        flip = rng.choice(data_pos, size=int(0.015 * data_pos.size), replace=False)
        noisy[flip] ^= 1
        if decode_frame(noisy)["bytes"] == frame.tobytes():
            corrected += 1
    return {"trials": trials, "clean_ok": clean_ok, "error_corrected": corrected}


if __name__ == "__main__":
    print(_selftest())
