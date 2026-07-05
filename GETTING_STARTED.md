# Getting started

You do not need any experience with radios or software to try this, and it is
free of charge.

## Try it now, with no radio

1. Open `LBandScope`.
2. Leave the channel and receiver set to Demo.
3. Click Start. Sample messages appear in the table.

Nothing to install, nothing to configure.

## Moving to a real receiver

Three steps below are hardware and Windows tasks that an application cannot do on
your behalf. The program tells you when they are needed; Help > Setup a receiver
repeats them.

### What you need
- An RTL-SDR USB receiver (the RTL-SDR Blog V3 is a common choice).
- An L-band patch antenna, often sold with the receiver.

### One-time setup (about ten minutes)
1. Connect the receiver to a USB port.
2. Install the USB driver with Zadig (<https://zadig.akeo.ie>). In Zadig:
   Options > List All Devices, select "Bulk-In, Interface (Interface 0)",
   choose WinUSB, and click Replace Driver.
3. Install receiver support so the application can see the device:
   `pip install pyrtlsdr`.
4. Place the patch antenna with a clear view of the sky toward the Inmarsat
   satellite for your region. It is geostationary, so once aimed it stays put.

### Then
1. Open `LBandScope`.
2. Click Refresh; the receiver appears in the list.
3. Choose a channel and the receiver, and click Start.

## What is in the download
- `LBandScope.exe` — the application. This is the one to open.
- `lbandscope-cli.exe` — the same functionality as a command-line tool, for
  scripting.

No Python, compilers, or other dependencies are required to run them.
