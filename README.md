# LBandScope

A decoder and monitor for Inmarsat L-band signals (Aero and STD-C/EGC), with a
point-and-click desktop application and a command-line tool.

This software is free of charge and always will be. It exists to make these
signals easier for other people to receive. It is licensed under the GNU General
Public License v3 (see `LICENSE`), which keeps it free to use, study, and share.

## What it does

- Receives IQ from an RTL-SDR or any SoapySDR device, a file, a pipe, or a TCP
  stream.
- Down-converts and resamples an arbitrary device rate to the channel of
  interest, so the decoder does not depend on the SDR's native sample rate.
- Detects and decodes frames continuously, reporting each with a timestamp.
- Shows a live spectrum waterfall, so signal presence is visible at a glance.
- Extracts positions from position-bearing messages and plots them on a map.
- Exports decoded data to CSV, GeoJSON, and KML for use in other tools.
- Runs a full offline demonstration with no radio attached.

## Quick start

You do not need to install anything or connect a radio to try it.

1. Run the application (`LBandScope`, or `python -m lbandscope`).
2. Leave the channel and receiver set to Demo.
3. Click Start. Decoded sample messages appear in the table.

## Using a real receiver

Three steps here are hardware and operating-system tasks that no application can
perform for you. The program detects when they are needed and walks you through
them (Help > Setup a receiver):

1. Connect an RTL-SDR and install its USB driver with Zadig
   (<https://zadig.akeo.ie>).
2. Install receiver support: `pip install pyrtlsdr` (or SoapySDR with its RTL-SDR
   module).
3. Fit an L-band patch antenna with a clear view of the sky toward the Inmarsat
   satellite for your region.

Then open the application, click Refresh, select the receiver and a channel, and
click Start.

## Running from source

Requires Python 3.9+ and NumPy.

```
pip install numpy
python -m lbandscope            # graphical application
python -m lbandscope doctor     # report SDR support and connected receivers
python -m lbandscope selftest   # verify the receiver with no radio
```

Command-line decoding from a capture or a live device:

```
# a raw capture (rtl_sdr uint8, or cf32 / cs16)
python -m lbandscope decode --source file --path capture.cf32 --fmt cf32

# a live pipe from another tool
rtl_sdr -f 1545000000 -s 2048000 - | python -m lbandscope decode --source stdin --fmt cu8

# a wider capture, down-converted to the channel
python -m lbandscope decode --source file --path wide.cf32 \
    --fs 2048000 --tune 300000 --baud 10500 --sps 8
```

Each decoded frame is printed as one JSON object.

## Prebuilt Windows binaries (optional)

For anyone who would rather not set up Python, prebuilt Windows executables are
provided under Releases:

- `LBandScope.exe` — the graphical application.
- `lbandscope-cli.exe` — the command-line tool.

Each is a single self-contained file; no Python or other dependencies are needed
on the target machine. They are optional and offer nothing the source does not.

## Building the binaries yourself

```
pip install pyinstaller numpy
powershell -ExecutionPolicy Bypass -File build.ps1
```

This produces both executables in `dist/` and runs the self-test as a release
check.

## How it works

The receiver chain is: squaring-loop carrier-frequency estimation, FFT
cross-correlation against the frame preamble with level-normalized detection,
sub-sample timing by parabolic interpolation, preamble-aided residual-frequency
and phase correction, coherent detection, descrambling, and CRC validation.
Detection runs over a continuous buffer and recovers every frame, streaming with
carry-over so frames that straddle block boundaries are not lost. `EVALUATION.md`
records the measured behavior.

## Scope

The reusable receiver stages and the front end (device I/O, channelizer, stream
framing, application) are in place and tested. The self-test frame format
validates the chain end to end without hardware. Waveform-specific demodulators
for the individual Inmarsat modes are the next addition; `channelize.rrc`
provides the matched filter they require.

## License

GNU General Public License v3.0. See `LICENSE`.
