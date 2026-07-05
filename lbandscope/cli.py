"""Command-line interface.

    lbandscope-cli                 open the graphical application (default)
    lbandscope-cli doctor          report SDR support and connected receivers
    lbandscope-cli selftest        run the receiver self-test (no radio)
    lbandscope-cli decode ...      decode IQ from a file, pipe, TCP, or SoapySDR
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import __version__, channelize, dsp, iqsource


def _run_selftest() -> int:
    cases = [
        (b"AES:4B1A2C ACARS/ADS-C POS N37.46 E013.22", 15.0, 3e-4, 1),
        (bytes(range(255)), 9.0, 1.2e-3, 7),
        (b"STD-C EGC MSG 12/34 SAR broadcast test", 13.0, 5e-4, 3),
    ]
    ok_all = True
    for i, (payload, snr, cfo, seed) in enumerate(cases):
        tx = dsp.modulate(payload)
        rx = dsp.apply_channel(tx, snr_db=snr, cfo=cfo, phase=1.1,
                               rng=np.random.default_rng(seed))
        got, ok = dsp.demodulate(rx)
        good = ok and got == payload
        ok_all &= good
        print(f"[{'PASS' if good else 'FAIL'}] case {i}: "
              f"{len(payload)}B  snr={snr}dB  cfo={cfo:+.1e}")

    rng = np.random.default_rng(77)
    payloads = [bytes(rng.integers(0, 256, int(rng.integers(1, 120))).tolist())
                for _ in range(12)]
    parts = []
    for p in payloads:
        parts.append(dsp.modulate(p))
        parts.append(0.01 * (rng.standard_normal(40) + 1j * rng.standard_normal(40)))
    rx = dsp.apply_channel(np.concatenate(parts), snr_db=13.0, cfo=6e-4, phase=0.9,
                           rng=np.random.default_rng(5))
    got = [f["payload"] for f in dsp.decode_frames(rx)]
    good = got == payloads
    ok_all &= good
    print(f"[{'PASS' if good else 'FAIL'}] multi-frame stream: "
          f"{sum(g == p for g, p in zip(got, payloads))}/{len(payloads)} recovered")

    rng = np.random.default_rng(1234)
    fails = 0
    for i in range(50):
        n = int(rng.integers(1, 220))
        payload = bytes(rng.integers(0, 256, n).tolist())
        tx = dsp.modulate(payload)
        rx = dsp.apply_channel(tx, snr_db=12.0,
                               cfo=float(rng.uniform(-1e-3, 1e-3)),
                               phase=float(rng.uniform(0, 6.28)),
                               rng=np.random.default_rng(i))
        gp, ok = dsp.demodulate(rx)
        if not (ok and gp == payload):
            fails += 1
    print(f"[{'PASS' if fails == 0 else 'FAIL'}] random sweep: {50 - fails}/50")
    ok_all &= fails == 0

    print("-" * 44)
    print("SELFTEST OK" if ok_all else "SELFTEST FAILED")
    return 0 if ok_all else 1


def _run_decode(args) -> int:
    ddc_on = args.fs > 0 and args.baud > 0
    out_rate = args.baud * args.sps if ddc_on else 0.0

    src = iqsource.open_source(args)
    carry = np.empty(0, dtype=np.complex128)
    overlap = dsp.MAX_FRAME_SYMS * args.sps
    process_min = max(overlap * 2, 1 << 18)
    abs_base = 0
    n_frames = 0

    def emit(f, base):
        nonlocal n_frames
        n_frames += 1
        p = f["payload"]
        print(json.dumps({"sample": base + f["sample"], "len": len(p),
                          "hex": p.hex(), "text": p.decode("latin-1")}), flush=True)

    def process(buf, base, flush):
        frames = dsp.decode_frames(buf, args.sps)
        keep_from = 0 if flush else max(0, len(buf) - overlap)
        for f in frames:
            if flush or f["sample"] < keep_from:
                emit(f, base)
        drop = len(buf) if flush else keep_from
        return buf[drop:], base + drop

    for block in src:
        block = block.astype(np.complex128)
        if ddc_on:
            block = channelize.ddc(block, args.fs, args.tune, out_rate)
        carry = np.concatenate([carry, block])
        if len(carry) >= process_min:
            carry, abs_base = process(carry, abs_base, flush=False)

    process(carry, abs_base, flush=True)
    print(f"# decoded {n_frames} frame(s)", file=sys.stderr)
    return 0


def _run_doctor() -> int:
    from . import sdr
    env = sdr.check_environment()
    print("Receiver check")
    print("-" * 40)
    print(f"  {env['summary']}")
    print(f"  SDR support: {', '.join(env['backends']) or 'none installed'}")
    print(f"  receivers connected: {len(env['devices'])}")
    for d in env["devices"]:
        print(f"    - {d['label']} ({d['backend']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lbandscope-cli",
                                description="Inmarsat L-band decoder")
    p.add_argument("--version", action="version", version=f"lbandscope {__version__}")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("gui", help="open the graphical application (default)")
    sub.add_parser("doctor", help="report SDR support and connected receivers")
    sub.add_parser("selftest", help="run the receiver self-test (no radio)")

    d = sub.add_parser("decode", help="decode IQ from a source")
    d.add_argument("--source", choices=["file", "stdin", "tcp", "soapy"], required=True)
    d.add_argument("--path", default="", help="file path, or host:port for tcp")
    d.add_argument("--fmt", choices=["cf32", "cs16", "cu8"], default="cf32")
    d.add_argument("--block", type=int, default=262144)
    d.add_argument("--sps", type=int, default=8, help="samples per symbol")
    d.add_argument("--fs", type=float, default=0.0, help="input sample rate (Hz); enables DDC")
    d.add_argument("--tune", type=float, default=0.0, help="channel offset from center (Hz)")
    d.add_argument("--baud", type=float, default=0.0, help="symbol rate (Hz); enables DDC")
    d.add_argument("--driver", default="rtlsdr", help="SoapySDR driver key")
    d.add_argument("--freq", type=float, default=1545e6)
    d.add_argument("--rate", type=float, default=2.048e6)
    d.add_argument("--gain", type=float, default=40.0)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd in (None, "gui"):
        from . import gui
        return gui.main()
    if args.cmd == "doctor":
        return _run_doctor()
    if args.cmd == "selftest":
        return _run_selftest()
    if args.cmd == "decode":
        return _run_decode(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
