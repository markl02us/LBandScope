"""
End-to-end test harness.

Two levels:
  1. loopback  -- synthesize a BPSK signal with a KNOWN payload, push it through
                  a realistic channel (CFO + phase + AWGN), decode it, assert the
                  payload comes back byte-exact with a valid CRC. This proves the
                  receiver chain (CFO -> sync -> timing -> descramble -> CRC) and
                  runs with no hardware.
  2. file I/O  -- same, but the signal is written to a .cf32 file and re-read
                  through the universal IQ input layer, proving the ingest path
                  that a real SDR / GNU Radio pipe would use.

Run:  python -m pytest tests -q        (or)   python tests/test_e2e.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lbandscope import (dsp, iqsource, channelize, demo, sdr, presets,  # noqa: E402
                        spectrum, messages, frontend, stdc)


def _roundtrip(payload: bytes, snr_db: float, cfo: float, seed_ch: int) -> bytes:
    tx = dsp.modulate(payload)
    rx = dsp.apply_channel(tx, snr_db=snr_db, cfo=cfo, phase=1.3,
                           rng=np.random.default_rng(seed_ch))
    got, ok = dsp.demodulate(rx)
    assert ok, "CRC failed"
    return got


def test_loopback_clean():
    payload = b"AES:4B1A2C  ACARS/ADS-C  POS N37.46 E013.22"
    got = _roundtrip(payload, snr_db=15.0, cfo=3e-4, seed_ch=1)
    assert got == payload, (got, payload)


def test_loopback_stressed():
    # low SNR + larger carrier offset -- still must recover exactly
    payload = bytes(range(255))
    got = _roundtrip(payload, snr_db=9.0, cfo=1.2e-3, seed_ch=7)
    assert got == payload


def test_many_random_frames():
    rng = np.random.default_rng(1234)
    fails = 0
    for i in range(40):
        n = int(rng.integers(1, 200))
        payload = bytes(rng.integers(0, 256, n).tolist())
        tx = dsp.modulate(payload)
        rx = dsp.apply_channel(tx, snr_db=12.0, cfo=float(rng.uniform(-1e-3, 1e-3)),
                               phase=float(rng.uniform(0, 2 * np.pi)),
                               rng=np.random.default_rng(i))
        got, ok = dsp.demodulate(rx)
        if not (ok and got == payload):
            fails += 1
    assert fails == 0, f"{fails}/40 frames failed"


def test_file_ingest_path(tmp_path=None):
    import tempfile
    payload = b"STD-C EGC  MSG 12/34  SAR broadcast test"
    tx = dsp.modulate(payload)
    rx = dsp.apply_channel(tx, snr_db=13.0, cfo=5e-4, phase=0.4,
                           rng=np.random.default_rng(3))
    d = tmp_path or tempfile.mkdtemp()
    path = os.path.join(str(d), "capture.cf32")
    inter = np.empty(rx.size * 2, dtype=np.float32)
    inter[0::2] = rx.real
    inter[1::2] = rx.imag
    inter.tofile(path)

    blocks = list(iqsource.from_file(path, fmt="cf32", block=1 << 20))
    stream = np.concatenate(blocks).astype(np.complex128)
    got, ok = dsp.demodulate(stream)
    assert ok and got == payload, (ok, got[:16])


def test_multiframe_stream():
    """A continuous stream of many back-to-back frames -> decode_frames must
    recover EVERY one, in order. This is the real-world case the old single-frame
    demodulator could not handle."""
    rng = np.random.default_rng(77)
    payloads = [bytes(rng.integers(0, 256, int(rng.integers(1, 120))).tolist())
                for _ in range(12)]
    parts = []
    for p in payloads:
        parts.append(dsp.modulate(p))
        parts.append(0.01 * (rng.standard_normal(40) + 1j * rng.standard_normal(40)))
    stream = np.concatenate(parts)
    rx = dsp.apply_channel(stream, snr_db=13.0, cfo=6e-4, phase=0.9,
                           rng=np.random.default_rng(5))
    frames = dsp.decode_frames(rx)
    got = [f["payload"] for f in frames]
    assert got == payloads, f"recovered {len(got)}/{len(payloads)}"


def test_ddc_any_sample_rate():
    """Simulate an SDR capturing the channel off-center at a higher sample rate,
    then down-convert with the channelizer and decode -> proves 'any SDR / any
    rate' works."""
    pay = b"DDC any-SDR path N37.46 E013.22"
    tx_hi = dsp.modulate(pay, sps=32)                 # as if captured 4x oversampled
    f = 0.12                                          # channel offset (cyc/sample)
    n = np.arange(len(tx_hi))
    hi = tx_hi * np.exp(1j * 2 * np.pi * f * n)
    hi = dsp.apply_channel(hi, snr_db=20.0, cfo=0.0, phase=0.3,
                           rng=np.random.default_rng(9))
    bb = channelize.ddc(hi, fs=1.0, f_offset=f, out_rate=0.25)   # -> sps 32->8
    got, ok = dsp.demodulate(bb, sps=8)
    assert ok and got == pay, (ok, got)


def test_presets_valid():
    assert presets.names()[0].startswith("Demo")
    for p in presets.PRESETS:
        assert "name" in p and "kind" in p
        if p["kind"] == "inmarsat":
            assert p["freq"] > 1.5e9 and p["baud"] > 0


def test_demo_mode_decodes():
    """Demo mode must actually produce decoded messages -- the beginner's first
    win, with no radio."""
    seen = []
    for block in demo.demo_iq_blocks(n_blocks=3, rng=np.random.default_rng(0)):
        for f in dsp.decode_frames(block):
            seen.append(f["payload"])
    assert len(seen) >= 4, f"only {len(seen)} demo frames decoded"
    assert all(m in demo.DEMO_MESSAGES for m in seen)


def test_doctor_no_crash_without_backend():
    """With no SDR driver installed (the new-user state), the doctor must return
    friendly guidance, never raise."""
    env = sdr.check_environment()
    assert env["state"] in ("ready", "no_device", "no_backend")
    assert isinstance(env["summary"], str) and env["summary"]
    assert isinstance(sdr.list_devices(), list)


def test_spectrum_locates_tone():
    x = np.exp(1j * 2 * np.pi * 0.1 * np.arange(4096))     # tone at +0.1 cyc/sample
    s = spectrum.spectrum_db(x, 256)
    assert len(s) == 256
    # peak sits right of DC-center (128) by ~0.1*256 = 25.6 bins
    assert 150 < int(np.argmax(s)) < 159
    rgb = spectrum.colorize(np.tile(s, (10, 1)))
    assert rgb.shape == (10, 256, 3) and rgb.dtype == np.uint8
    assert spectrum.to_ppm(rgb).startswith(b"P6 256 10 255")


def test_message_parse_and_export():
    recs = [messages.make_record(t, "12:00:00") for t in (
        "AES 4B1A2C AIRLINE OPS FLT AZA123 POS N41.9 E012.5 FL370",
        "STD-C SHIP MV SICILIA  POS S12.3 W045.6  SOG 9kn",
        "STD-C EGC NAVAREA III gale warning")]
    assert recs[0]["kind"] == "Aero" and recs[0]["id"] == "AZA123"
    assert abs(recs[0]["lat"] - 41.9) < 1e-6 and abs(recs[0]["lon"] - 12.5) < 1e-6
    assert recs[1]["lat"] < 0 and recs[1]["lon"] < 0          # S/W signs
    assert recs[2]["lat"] is None                            # no position
    assert "AZA123" in messages.to_csv(recs)
    assert '"type": "FeatureCollection"' in messages.to_geojson(recs)
    gj = messages.to_geojson(recs)
    assert gj.count('"Point"') == 2                          # only the two with fixes
    assert "<Placemark>" in messages.to_kml(recs)


def test_constellation_symbols():
    rx = dsp.apply_channel(dsp.modulate(b"constellation check"), snr_db=20,
                           cfo=3e-4, phase=0.4, rng=np.random.default_rng(0))
    fr = dsp.decode_frames(rx, with_symbols=True, max_frames=1)
    assert fr and "symbols" in fr[0]
    pre = fr[0]["symbols"][:64]                       # preamble symbols
    assert np.iscomplexobj(pre)
    # BPSK after recovery: energy on the real axis, small quadrature component
    assert np.mean(np.abs(pre.imag)) < np.mean(np.abs(pre.real))
    pts = dsp.constellation(rx)
    assert 0 < len(pts) <= 800


def test_find_peak_offset():
    fs = 2.048e6
    x = np.exp(1j * 2 * np.pi * 0.1 * np.arange(8192))   # +0.1 cyc/sample carrier
    off, prom = frontend.find_peak_offset(spectrum.spectrum_db(x, 256), fs)
    assert abs(off - 0.1 * fs) < 0.02 * fs and prom > 10


def test_frontend_conditioning():
    # DC removal
    x = np.exp(1j * 2 * np.pi * 0.05 * np.arange(4096)) + (0.4 + 0.3j)
    assert abs(frontend.remove_dc(x).mean()) < 1e-9

    # IQ imbalance correction improves image rejection
    f, n = 0.1, np.arange(8192)
    tone = np.exp(1j * 2 * np.pi * f * n)
    phi, g = 0.15, 1.25
    y = tone.real + 1j * (g * (tone.imag * np.cos(phi) + tone.real * np.sin(phi)))

    def irr(z):
        S = np.abs(np.fft.fft(z))
        N = len(z)
        return 20 * np.log10(S[int(round(f * N))] / S[int(round((1 - f) * N))])

    assert irr(frontend.correct_iq_imbalance(y)) - irr(y) > 30.0

    # conditioning does not break a real decode
    rx = dsp.apply_channel(dsp.modulate(b"conditioned frame"), snr_db=12,
                           cfo=5e-4, phase=0.5, rng=np.random.default_rng(0))
    got, ok = dsp.demodulate(frontend.precondition(rx))
    assert ok and got == b"conditioned frame"

    assert isinstance(frontend.signal_level_db(tone), float)
    assert frontend.quality_db(spectrum.spectrum_db(tone, 256)) > 10


def test_stdc_chain_roundtrip():
    """The Inmarsat-C receive chain (UW sync, depermute, deinterleave, K=7
    Viterbi, descramble) must be exactly inverse to the transmit chain, and the
    FEC must correct a realistic symbol-error rate."""
    rng = np.random.default_rng(11)
    frame = rng.integers(0, 256, stdc.INFO_BYTES, dtype=np.uint8).tobytes()
    sym = stdc.encode_frame(frame)
    assert sym.size == stdc.FRAME_SYMBOLS
    assert stdc.decode_frame(sym)["bytes"] == frame

    # frame sync locates an embedded frame at the right offset and polarity
    stream = np.concatenate([rng.integers(0, 2, 400, dtype=np.uint8), sym,
                             rng.integers(0, 2, 200, dtype=np.uint8)])
    hits = stdc.find_uw(stream)
    assert hits and hits[0][0] == 400 and not hits[0][1]
    assert stdc.decode_frame(stream[400:400 + stdc.FRAME_SYMBOLS])["bytes"] == frame

    # Viterbi corrects 1% of the data-carrying symbols flipped
    noisy = sym.copy()
    flip = rng.choice(stdc._TX_POS, int(0.01 * stdc._TX_POS.size), replace=False)
    noisy[flip] ^= 1
    assert stdc.decode_frame(noisy)["bytes"] == frame


def test_gui_constructs():
    """The window must build without error (skipped if no display available)."""
    import tkinter as tk
    from lbandscope import gui
    try:
        app = gui.App()
    except tk.TclError as e:
        print(f"   (skip GUI construct: no display: {e})")
        return
    try:
        app.refresh_devices()
        assert app.device["values"][0].startswith("Demo")
    finally:
        app.destroy()


if __name__ == "__main__":
    tests = [test_loopback_clean, test_loopback_stressed,
             test_many_random_frames, test_file_ingest_path,
             test_multiframe_stream, test_ddc_any_sample_rate,
             test_presets_valid, test_demo_mode_decodes,
             test_doctor_no_crash_without_backend,
             test_spectrum_locates_tone, test_message_parse_and_export,
             test_constellation_symbols, test_find_peak_offset,
             test_frontend_conditioning, test_stdc_chain_roundtrip,
             test_gui_constructs]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print("-" * 40)
    print("ALL PASS" if failed == 0 else f"{failed} FAILED")
    sys.exit(1 if failed else 0)
