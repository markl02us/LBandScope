"""End-to-end evaluation of the Inmarsat-C (STD-C) receiver.

No off-air captures are published for this signal, so this reproduces a faithful
downlink instead: real unique word, permutation, interleaver, K=7 convolutional
code, and scrambler (the same bytes a real terminal transmits), root-raised-cosine
shaped, then impaired with additive noise, a carrier offset, and a sample-clock
error. The whole receiver -- demodulator, frame sync, Viterbi, descrambler, and
message parser -- is then run against it and scored.

Run:  python tests/evaluate_stdc.py

"SNR" here is broadband, measured at 8 samples per symbol; the matched filter adds
about 9 dB of processing gain on top, so the per-symbol Es/N0 is correspondingly
higher.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lbandscope import stdc, stdc_demod, stdc_parser  # noqa: E402

FS = int(stdc_demod.SYMBOL_RATE * 8)

# A realistic mix of EGC safety traffic: (messageType, priority, text).
TRAFFIC = [
    (0x31, 2, "NAVAREA III 042/26 GALE WARNING STRAIT OF SICILY WIND SW 30KT POS N37.5 E013.2"),
    (0x14, 3, "DISTRESS RELAY MAYDAY MV EXAMPLE TAKING WATER POS N36.8 E015.1 SOULS 12"),
    (0x13, 1, "SECURITE COASTAL WARNING UNLIT BUOY ADRIFT POS N38.1 E012.4"),
    (0x24, 2, "METAREA III FORECAST TYRRHENIAN SEA WIND NW 20KT SEA MODERATE POS N39.0 E013.0"),
    (0x04, 2, "SAR COORD SEARCH AREA CENTRE POS N37.0 E014.0 RADIUS 20NM VESSELS ASSIST"),
]


def make_signal(messages, snr_db, cfo_hz=70.0, clock_ppm=0.0, phase=0.6, seed=0):
    """Build a continuous single-carrier STD-C downlink of `messages`."""
    rng = np.random.default_rng(seed)
    syms = [stdc.encode_frame(stdc_parser.build_frame(
        [stdc_parser.build_egc(mt, prio, text, message_id=0x1000 + i)]))
        for i, (mt, prio, text) in enumerate(messages)]
    tx = stdc_demod.modulate_iq(np.concatenate(syms))
    if clock_ppm:
        m = int(len(tx) / (1 + clock_ppm / 1e6))
        pos = np.arange(m) * (1 + clock_ppm / 1e6)
        xp = np.arange(len(tx))
        tx = np.interp(pos, xp, tx.real) + 1j * np.interp(pos, xp, tx.imag)
    n = np.arange(len(tx))
    rx = tx * np.exp(1j * (2 * np.pi * (cfo_hz / FS) * n + phase))
    p = np.mean(np.abs(tx) ** 2) / 10 ** (snr_db / 10)
    rx = rx + np.sqrt(p / 2) * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))
    return rx


def _decodes(snr, seed, soft, cfo=70.0, ppm=0.0):
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, stdc.INFO_BYTES, dtype=np.uint8).tobytes()
    # one random frame is the strictest per-bit test
    syms = stdc.encode_frame(frame)
    tx = stdc_demod.modulate_iq(syms)
    if ppm:
        m = int(len(tx) / (1 + ppm / 1e6))
        pos = np.arange(m) * (1 + ppm / 1e6)
        xp = np.arange(len(tx))
        tx = np.interp(pos, xp, tx.real) + 1j * np.interp(pos, xp, tx.imag)
    n = np.arange(len(tx))
    rx = tx * np.exp(1j * (2 * np.pi * (cfo / FS) * n + 0.6))
    p = np.mean(np.abs(tx) ** 2) / 10 ** (snr / 10)
    rx = rx + np.sqrt(p / 2) * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))
    f = stdc_demod.receive(rx, FS, soft=soft)
    return bool(f) and f[0]["bytes"] == frame


def sensitivity(trials=40):
    print("1. Sensitivity -- frame decode rate vs SNR (hard vs soft Viterbi)")
    print("   bbSNR   hard      soft")
    for snr in (-4, -5, -6, -7, -8, -9):
        h = sum(_decodes(snr, s, False) for s in range(trials))
        so = sum(_decodes(snr, s, True) for s in range(trials))
        print(f"   {snr:>3} dB  {h:>2}/{trials}    {so:>2}/{trials}")


def carrier_tolerance(trials=30):
    print("\n2. Carrier-offset tolerance (soft, bbSNR = -4 dB)")
    print("   offset     decoded")
    for cfo in (0, 100, 200, 300, 400):
        ok = sum(_decodes(-4, s, True, cfo=float(cfo)) for s in range(trials))
        print(f"   {cfo:>4} Hz    {ok:>2}/{trials}")


def clock_tolerance(trials=30):
    print("\n3. Sample-clock tolerance (soft, bbSNR = -4 dB)")
    print("   error      decoded")
    for ppm in (0, 10, 20, 40, 80, 120):
        ok = sum(_decodes(-4, s, True, ppm=float(ppm)) for s in range(trials))
        print(f"   {ppm:>3} ppm    {ok:>2}/{trials}")


def end_to_end():
    print("\n4. End-to-end message integrity (continuous 5-message downlink)")
    rx = make_signal(TRAFFIC, snr_db=0.0, cfo_hz=180.0, clock_ppm=15.0, seed=3)
    frames = stdc_demod.receive(rx, FS)
    got = []
    for fr in frames:
        got += stdc_parser.messages(stdc_parser.parse_frame(fr["bytes"]))
    texts = {m["text"] for m in got}
    recovered = sum(any(text in t for t in texts) for _, _, text in TRAFFIC)
    distress = any(m.get("isDistress") for m in got)
    with_pos = sum(1 for m in got if "POS N" in m.get("text", ""))
    print(f"   messages transmitted: {len(TRAFFIC)}")
    print(f"   messages recovered:   {recovered}/{len(TRAFFIC)}")
    print(f"   distress flagged:     {distress}")
    print(f"   messages with a position fix: {with_pos}")
    for m in got:
        tag = "DISTRESS" if m.get("isDistress") else m.get("priorityText", "")
        print(f"     [{tag:>8}] {m['text'][:58]}")


def throughput():
    print("\n5. Throughput")
    rx = make_signal(TRAFFIC, snr_db=6.0, seed=1)
    dur = len(rx) / FS
    t0 = time.perf_counter()
    frames = stdc_demod.receive(rx, FS)
    dt = time.perf_counter() - t0
    print(f"   {len(rx)} samples ({dur:.1f} s of signal), {len(frames)} frames")
    print(f"   decoded in {dt:.2f} s  ({dur / dt:.1f}x real time)")


if __name__ == "__main__":
    print("=" * 60)
    print("Inmarsat-C (STD-C) end-to-end evaluation")
    print("=" * 60)
    sensitivity()
    carrier_tolerance()
    clock_tolerance()
    end_to_end()
    throughput()
    print("=" * 60)
