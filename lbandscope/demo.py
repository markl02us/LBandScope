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
