"""IQ input from files, pipes, TCP, or a SoapySDR device.

All sources yield blocks of complex64. File/pipe/TCP formats:
    cf32  interleaved float32 I,Q   (GNU Radio)
    cs16  interleaved int16   I,Q
    cu8   interleaved uint8    I,Q   (rtl_sdr)
"""
from __future__ import annotations

import socket
import sys
from typing import Iterator

import numpy as np

_DTYPE = {"cf32": np.float32, "cs16": np.int16, "cu8": np.uint8}


def _convert(raw: np.ndarray, fmt: str) -> np.ndarray:
    if fmt == "cf32":
        f = raw.astype(np.float32)
    elif fmt == "cs16":
        f = raw.astype(np.float32) / 32768.0
    elif fmt == "cu8":
        f = (raw.astype(np.float32) - 127.5) / 127.5
    else:
        raise ValueError(f"unknown IQ format: {fmt}")
    return (f[0::2] + 1j * f[1::2]).astype(np.complex64)


def from_file(path: str, fmt: str = "cf32", block: int = 262144) -> Iterator[np.ndarray]:
    dt = np.dtype(_DTYPE[fmt])
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(block * 2 * dt.itemsize)
            if not buf:
                break
            raw = np.frombuffer(buf, dtype=dt)
            if raw.size >= 2:
                yield _convert(raw[: (raw.size // 2) * 2], fmt)


def from_stdin(fmt: str = "cf32", block: int = 262144) -> Iterator[np.ndarray]:
    dt = np.dtype(_DTYPE[fmt])
    stream = sys.stdin.buffer
    while True:
        buf = stream.read(block * 2 * dt.itemsize)
        if not buf:
            break
        raw = np.frombuffer(buf, dtype=dt)
        if raw.size >= 2:
            yield _convert(raw[: (raw.size // 2) * 2], fmt)


def from_tcp(host: str, port: int, fmt: str = "cu8", block: int = 262144) -> Iterator[np.ndarray]:
    dt = np.dtype(_DTYPE[fmt])
    need = block * 2 * dt.itemsize
    with socket.create_connection((host, port)) as s:
        buf = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
            while len(buf) >= need:
                raw = np.frombuffer(buf[:need], dtype=dt)
                buf = buf[need:]
                yield _convert(raw, fmt)


def from_soapy(driver: str, freq: float, rate: float, gain: float = 40.0,
               block: int = 262144) -> Iterator[np.ndarray]:
    """Stream from any SoapySDR device. Imported lazily so the file/pipe/TCP
    paths work without SoapySDR installed."""
    try:
        import SoapySDR
        from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
    except Exception as e:  # pragma: no cover - hardware path
        raise RuntimeError(
            "SoapySDR is not installed; use --source file/stdin/tcp instead."
        ) from e

    sdr = SoapySDR.Device(dict(driver=driver))
    sdr.setSampleRate(SOAPY_SDR_RX, 0, rate)
    sdr.setFrequency(SOAPY_SDR_RX, 0, freq)
    sdr.setGain(SOAPY_SDR_RX, 0, gain)
    st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(st)
    buff = np.empty(block, np.complex64)
    try:
        while True:
            sr = sdr.readStream(st, [buff], block)
            if sr.ret > 0:
                yield buff[: sr.ret].copy()
    finally:  # pragma: no cover - hardware path
        sdr.deactivateStream(st)
        sdr.closeStream(st)


def open_source(args) -> Iterator[np.ndarray]:
    if args.source == "file":
        return from_file(args.path, args.fmt, args.block)
    if args.source == "stdin":
        return from_stdin(args.fmt, args.block)
    if args.source == "tcp":
        host, _, port = args.path.partition(":")
        return from_tcp(host, int(port), args.fmt, args.block)
    if args.source == "soapy":
        return from_soapy(args.driver, args.freq, args.rate, args.gain, args.block)
    raise ValueError(f"unknown source: {args.source}")
