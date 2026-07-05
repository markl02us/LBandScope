"""SDR device discovery and streaming.

Backends are detected and imported lazily; with none installed the functions
return empty results rather than raising, so the rest of the application runs.

    pyrtlsdr  RTL-SDR dongles
    SoapySDR  RTL-SDR, Airspy, HackRF, SDRplay, and others
"""
from __future__ import annotations

import importlib.util

import numpy as np


def available_backends():
    found = []
    if importlib.util.find_spec("rtlsdr") is not None:
        found.append("rtlsdr")
    if importlib.util.find_spec("SoapySDR") is not None:
        found.append("soapy")
    return found


def list_devices():
    """Return [{'backend', 'id', 'label'}] for connected devices."""
    devs = []
    try:
        from rtlsdr import RtlSdr
        for i in range(RtlSdr.get_device_count()):
            try:
                name = RtlSdr.get_device_name(i)
            except Exception:
                name = "RTL-SDR"
            devs.append({"backend": "rtlsdr", "id": i, "label": f"{name} #{i}"})
    except Exception:
        pass
    try:
        import SoapySDR
        for d in SoapySDR.Device.enumerate():
            dd = dict(d)
            devs.append({"backend": "soapy", "id": dd,
                         "label": dd.get("label", dd.get("driver", "SoapySDR"))})
    except Exception:
        pass
    return devs


def check_environment():
    """Summarize backend/device availability for the UI and CLI."""
    backends = available_backends()
    devices = list_devices()
    if devices:
        state = "ready"
        summary = f"Radio detected: {devices[0]['label']}."
    elif backends:
        state = "no_device"
        summary = "SDR support is installed but no radio is connected."
    else:
        state = "no_backend"
        summary = "No SDR support detected. Demo mode is available now."
    return {"state": state, "summary": summary,
            "backends": backends, "devices": devices}


def stream(device, cfg, block=262144):
    """Yield complex64 IQ blocks from a connected device."""
    if device["backend"] == "rtlsdr":
        from rtlsdr import RtlSdr
        sdr = RtlSdr(device_index=int(device["id"]))
        sdr.sample_rate = cfg["rate"]
        sdr.center_freq = cfg["freq"]
        try:
            sdr.gain = "auto"
        except Exception:
            pass
        if cfg.get("bias_tee"):
            # Powers the antenna's LNA. Not all librtlsdr builds expose this.
            try:
                sdr.set_bias_tee(True)
            except Exception:
                pass
        try:
            while True:
                yield np.asarray(sdr.read_samples(block), dtype=np.complex64)
        finally:
            sdr.close()
    elif device["backend"] == "soapy":
        import SoapySDR
        from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
        dev = SoapySDR.Device(device["id"])
        dev.setSampleRate(SOAPY_SDR_RX, 0, cfg["rate"])
        dev.setFrequency(SOAPY_SDR_RX, 0, cfg["freq"])
        st = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        dev.activateStream(st)
        buff = np.empty(block, np.complex64)
        try:
            while True:
                sr = dev.readStream(st, [buff], block)
                if sr.ret > 0:
                    yield buff[:sr.ret].copy()
        finally:
            dev.deactivateStream(st)
            dev.closeStream(st)
    else:
        raise RuntimeError(f"unknown backend: {device['backend']}")
