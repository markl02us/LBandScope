"""
Quantitative evaluation of the receiver chain -- find the limits, not just "PASS".

Reports:
  1. Sensitivity      FER vs SNR waterfall -> usable threshold
  2. CFO capture      max carrier offset before sync collapses
  3. Timing/clock     fractional-sample delay + sample-clock ppm offset
  4. Bandlimiting     real signals aren't rectangular -> lowpass the TX
  5. False alarm      pure noise in -> must decode ~0 frames (no hallucination)
  6. Ingest formats   cu8 (rtl_sdr native) + cs16 round-trips actually decode

Honest by design: negatives are reported, not hidden.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbandscope import dsp, iqsource  # noqa: E402

RNG = np.random.default_rng(20260704)


def _payload(n):
    return bytes(RNG.integers(0, 256, n).tolist())


def _fer(snr, cfo_max=8e-4, trials=200, plen=40, extra=None):
    """Frame-error-rate over `trials` random frames."""
    fails = 0
    for i in range(trials):
        pay = _payload(plen)
        tx = dsp.modulate(pay)
        cfo = float(RNG.uniform(-cfo_max, cfo_max))
        rx = dsp.apply_channel(tx, snr_db=snr, cfo=cfo,
                               phase=float(RNG.uniform(0, 6.28)),
                               rng=np.random.default_rng(i))
        if extra is not None:
            rx = extra(rx)
        got, ok = dsp.demodulate(rx)
        if not (ok and got == pay):
            fails += 1
    return fails / trials


def eval_sensitivity():
    print("\n== 1. Sensitivity (FER vs SNR, 200 frames/pt) ==")
    print(f"{'SNR dB':>7} {'FER':>9}  {'':<20}")
    thresh = None
    for snr in [3, 4, 5, 6, 7, 8, 9, 10, 12, 14]:
        fer = _fer(snr)
        bar = "#" * int(round(fer * 20))
        print(f"{snr:>7} {fer:>9.3f}  {bar:<20}")
        if thresh is None and fer <= 0.01:
            thresh = snr
    print(f"   -> usable threshold (FER<=1%): "
          f"{'%d dB' % thresh if thresh else 'NOT REACHED'}")


def eval_cfo():
    print("\n== 2. CFO capture range (SNR=12dB, 150 frames/pt) ==")
    print(f"{'cfo cyc/s':>10} {'FER':>9}")
    last_good = None
    for cfo in [1e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2]:
        # fixed-magnitude offset, random sign
        fails = 0
        for i in range(150):
            pay = _payload(40)
            tx = dsp.modulate(pay)
            s = 1 if i % 2 else -1
            rx = dsp.apply_channel(tx, snr_db=12, cfo=s * cfo,
                                   phase=float(RNG.uniform(0, 6.28)),
                                   rng=np.random.default_rng(i))
            got, ok = dsp.demodulate(rx)
            if not (ok and got == pay):
                fails += 1
        fer = fails / 150
        print(f"{cfo:>10.1e} {fer:>9.3f}")
        if fer <= 0.01:
            last_good = cfo
    print(f"   -> holds lock to +/-{last_good:.1e} cyc/sample"
          if last_good else "   -> FAILS even at smallest offset")


def _frac_delay(x, d):
    """Delay by fractional sample d via FFT linear phase."""
    n = len(x)
    f = np.fft.fftfreq(n)
    return np.fft.ifft(np.fft.fft(x) * np.exp(-1j * 2 * np.pi * f * d))


def _clock_offset(x, ppm):
    """Resample as if RX clock differs by `ppm` (sample-timing drift)."""
    n = len(x)
    new_n = int(round(n * (1 + ppm * 1e-6)))
    xi = np.arange(n)
    xo = np.linspace(0, n - 1, new_n)
    return np.interp(xo, xi, x.real) + 1j * np.interp(xo, xi, x.imag)


def _leadin_fracdelay(r, d):
    # realistic capture: quiet guard samples on BOTH sides of the frame (a real
    # stream is continuous), then a fractional-sample delay. Without trailing
    # guard, a fractional start would push the final symbol off the buffer end.
    g = lambda s, k: 0.02 * (np.random.default_rng(s).standard_normal(k)
                             + 1j * np.random.default_rng(s + 1).standard_normal(k))
    return _frac_delay(np.concatenate([g(0, 37), r, g(4, 40)]), d)


def eval_timing():
    print("\n== 3. Timing robustness (SNR=12dB, 150 frames/pt) ==")
    fer_fd = _fer(12, trials=150, extra=lambda r: _leadin_fracdelay(r, 0.5))
    print(f"   fractional 0.5-sample delay : FER={fer_fd:.3f}  (with capture lead-in)")
    for ppm in [50, 200, 1000]:
        fer = _fer(12, trials=150, extra=lambda r, p=ppm: _clock_offset(r, p))
        print(f"   sample-clock {ppm:>5} ppm     : FER={fer:.3f}")
    print("   (no fractional timing-recovery loop yet -> watch the ppm rows)")


def eval_bandlimit():
    print("\n== 4. Bandlimiting (SNR=12dB, 150 frames) ==")

    def lp(r, taps):
        h = np.ones(taps) / taps
        return np.convolve(r, h, mode="same")

    for taps in [2, 3, 5]:
        fer = _fer(12, trials=150, extra=lambda r, t=taps: lp(r, t))
        print(f"   moving-avg lowpass {taps} taps : FER={fer:.3f}")
    print("   (rectangular-pulse assumption; real RRC needs a matched filter)")


def eval_false_alarm():
    print("\n== 5. False-alarm on pure noise (2000 trials) ==")
    hall = 0
    for i in range(2000):
        n = int(RNG.integers(600, 4000))
        noise = (RNG.standard_normal(n) + 1j * RNG.standard_normal(n))
        got, ok = dsp.demodulate(noise)
        if ok:
            hall += 1
    print(f"   spurious decodes: {hall}/2000  "
          f"(expect ~0; CRC-16 false-pass ~1.5e-5)")


def eval_ingest():
    print("\n== 6. Ingest-format round-trips (cu8 / cs16) ==")
    pay = b"rtl_sdr native format check N37.46 E013.22"
    tx = dsp.modulate(pay)
    rx = dsp.apply_channel(tx, snr_db=14, cfo=4e-4, phase=0.5,
                           rng=np.random.default_rng(99))
    import tempfile
    d = tempfile.mkdtemp()
    for fmt in ["cf32", "cs16", "cu8"]:
        path = os.path.join(d, f"cap.{fmt}")
        if fmt == "cf32":
            a = np.empty(rx.size * 2, np.float32)
            a[0::2], a[1::2] = rx.real, rx.imag
            a.astype(np.float32).tofile(path)
        elif fmt == "cs16":
            a = np.empty(rx.size * 2, np.int16)
            a[0::2] = np.clip(rx.real * 8000, -32768, 32767)
            a[1::2] = np.clip(rx.imag * 8000, -32768, 32767)
            a.tofile(path)
        else:  # cu8 : center 127.5, limited 8-bit dynamic range
            a = np.empty(rx.size * 2, np.uint8)
            a[0::2] = np.clip(rx.real * 40 + 127.5, 0, 255).astype(np.uint8)
            a[1::2] = np.clip(rx.imag * 40 + 127.5, 0, 255).astype(np.uint8)
            a.tofile(path)
        blocks = list(iqsource.from_file(path, fmt=fmt, block=1 << 20))
        stream = np.concatenate(blocks).astype(np.complex128)
        got, ok = dsp.demodulate(stream)
        good = ok and got == pay
        print(f"   {fmt:>4}: {'PASS' if good else 'FAIL'}  "
              f"({'exact' if got == pay else 'mismatch/none'})")


def eval_multiframe():
    print("\n== 7. Multi-frame stream recovery ==")
    for snr in [8, 10, 12]:
        rng = np.random.default_rng(int(snr))
        pays = [_payload(int(rng.integers(1, 120))) for _ in range(30)]
        parts = []
        for p in pays:
            parts.append(dsp.modulate(p))
            parts.append(0.01 * (rng.standard_normal(50) + 1j * rng.standard_normal(50)))
        rx = dsp.apply_channel(np.concatenate(parts), snr_db=snr, cfo=7e-4,
                               phase=1.0, rng=rng)
        got = [f["payload"] for f in dsp.decode_frames(rx)]
        n_ok = sum(1 for g, p in zip(got, pays) if g == p) if len(got) == len(pays) else \
            sum(1 for g in got if g in pays)
        print(f"   snr={snr}dB : {n_ok}/{len(pays)} frames recovered")


def eval_throughput():
    print("\n== 8. Throughput ==")
    import time
    from lbandscope import channelize
    rng = np.random.default_rng(0)

    # (a) decode_frames runs on the DECIMATED channel (post-DDC), not raw Msps
    parts = [dsp.apply_channel(dsp.modulate(_payload(60)), snr_db=15, cfo=4e-4,
                               phase=0.3, rng=np.random.default_rng(k))
             for k in range(6)]
    buf = np.concatenate([0.01 * (rng.standard_normal(150000)
                                  + 1j * rng.standard_normal(150000))]
                         + [x for p in parts for x in
                            (p, 0.01 * (rng.standard_normal(20000)
                                        + 1j * rng.standard_normal(20000)))])
    t0 = time.perf_counter(); dsp.decode_frames(buf); t_dec = time.perf_counter() - t0
    msps = len(buf) / t_dec / 1e6
    print(f"   decode_frames: {msps:.1f} Msamp/s on {len(buf)/1e6:.2f} Msamples")
    for name, ch_rate in [("Aero 10.5k baud x8", 84e3), ("STD-C 1200 baud x8", 9.6e3)]:
        print(f"     -> {name:22} ({ch_rate/1e3:5.1f} kSamp/s): "
              f"{msps*1e6/ch_rate:6.0f}x realtime")

    # (b) DDC front-end must keep up with the raw SDR rate
    raw = (rng.standard_normal(2_048_000) + 1j * rng.standard_normal(2_048_000)) * 0.1
    t0 = time.perf_counter()
    channelize.ddc(raw, fs=2.048e6, f_offset=3e5, out_rate=84e3)
    t_ddc = time.perf_counter() - t0
    print(f"   DDC front-end: {2.048/t_ddc:.1f}x realtime on RTL-SDR 2.048 Msps "
          f"({t_ddc*1e3:.0f} ms / 1 s of IQ)")


if __name__ == "__main__":
    eval_sensitivity()
    eval_cfo()
    eval_timing()
    eval_bandlimit()
    eval_false_alarm()
    eval_ingest()
    eval_multiframe()
    eval_throughput()
    print("\ndone.")
