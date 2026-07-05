"""Channel presets.

Starting points for the Inmarsat L-band downlink (1525-1559 MHz). Exact channel
frequencies vary by satellite and region.
"""
from __future__ import annotations

PRESETS = [
    {
        "name": "Demo (no radio required)",
        "kind": "demo",
        "blurb": "Exercise the decoder with sample traffic.",
    },
    {
        "name": "Inmarsat Aero",
        "kind": "inmarsat",
        "freq": 1545_000_000,
        "rate": 2_048_000,
        "baud": 10_500,
        "sps": 8,
        "blurb": "Aeronautical data and voice.",
    },
    {
        "name": "Inmarsat STD-C / EGC",
        "kind": "inmarsat",
        "freq": 1541_450_000,
        "rate": 2_048_000,
        "baud": 1_200,
        "sps": 8,
        "blurb": "Maritime messaging and safety broadcasts.",
    },
]


def by_name(name: str):
    for p in PRESETS:
        if p["name"] == name:
            return p
    return PRESETS[0]


def names():
    return [p["name"] for p in PRESETS]
