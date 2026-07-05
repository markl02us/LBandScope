"""Offline demonstration source.

Generates a stream of frames carrying sample message text and passes it through
the channel model, so the full detect/decode path can be exercised without a
radio. Message content is synthetic.
"""
from __future__ import annotations

import numpy as np

from . import dsp

SAMPLE_MESSAGES = [
    b"AES 4B1A2C  AIRLINE OPS  FLT AZA123  POS N41.9 E012.5  FL370",
    b"STD-C EGC  NAVAREA III  SECURITE  gale warning Ionian Sea",
    b"AES 3C22F1  ACARS  ETA LICJ 14:35Z  fuel 8.2t  souls 148",
    b"STD-C  SHIP MV SICILIA  POS N37.5 E013.2  SOG 12.4kn",
    b"AES 4B1A2C  ADS-C  WPT DITAK  next KONBA  M0.79",
    b"STD-C EGC  METAREA  forecast Strait of Sicily  wind SW 20kn",
]

# Backwards-compatible alias.
DEMO_MESSAGES = SAMPLE_MESSAGES


# EGC traffic for the STD-C demonstration: (messageType, priority, text).
# Positions are written so the map picks them up.
STDC_DEMO_MESSAGES = [
    (0x31, 2, "NAVAREA III 042/26 GALE WARNING STRAIT OF SICILY WIND SW 30KT POS N37.5 E013.2"),
    (0x14, 3, "DISTRESS RELAY MAYDAY MV EXAMPLE TAKING WATER POS N36.8 E015.1 SOULS 12"),
    (0x13, 1, "SECURITE COASTAL WARNING UNLIT BUOY ADRIFT POS N38.1 E012.4"),
    (0x24, 2, "METAREA III FORECAST TYRRHENIAN SEA WIND NW 20KT SEA MODERATE POS N39.0 E013.0"),
]


def stdc_demo_blocks(n_blocks: int = 10 ** 9, rng: np.random.Generator | None = None):
    """Yield chunks of a continuous Inmarsat-C downlink carrying sample EGC
    traffic, generated through the real STD-C transmit path (one carrier, as a
    real satellite) so the actual receiver decodes it frame by frame."""
    from . import stdc, stdc_demod, stdc_parser
    rng = rng or np.random.default_rng(0)
    fs = stdc_demod.SYMBOL_RATE * 8
    syms = [stdc.encode_frame(stdc_parser.build_frame(
        [stdc_parser.build_egc(mt, prio, text, message_id=0x1000 + i)]))
        for i, (mt, prio, text) in enumerate(STDC_DEMO_MESSAGES)]
    tx = stdc_demod.modulate_iq(np.concatenate(syms))
    n = np.arange(len(tx))
    rx = tx * np.exp(1j * (2 * np.pi * (60.0 / fs) * n + 0.4))
    p = np.mean(np.abs(tx) ** 2) / 10 ** (14 / 10)
    rx = (rx + np.sqrt(p / 2) * (rng.standard_normal(len(rx))
                                 + 1j * rng.standard_normal(len(rx)))).astype(np.complex64)
    chunk = len(tx) // len(syms)                    # about one frame per chunk
    pos = 0
    for _ in range(n_blocks):
        if pos + chunk > len(rx):
            pos = 0
        yield rx[pos:pos + chunk]
        pos += chunk


def demo_iq_blocks(n_blocks: int = 6, rng: np.random.Generator | None = None):
    """Yield IQ blocks, two frames per block, with a small inter-frame gap."""
    rng = rng or np.random.default_rng(0)
    idx = 0
    for _ in range(n_blocks):
        parts = []
        for _ in range(2):
            msg = SAMPLE_MESSAGES[idx % len(SAMPLE_MESSAGES)]
            idx += 1
            parts.append(dsp.modulate(msg))
            parts.append(0.01 * (rng.standard_normal(80) + 1j * rng.standard_normal(80)))
        block = dsp.apply_channel(np.concatenate(parts), snr_db=13.0,
                                  cfo=float(rng.uniform(-8e-4, 8e-4)),
                                  phase=float(rng.uniform(0, 6.28)), rng=rng)
        yield block.astype(np.complex64)
