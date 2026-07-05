# Receiver evaluation

Measured behavior of the receiver chain, produced by `python tests/evaluate.py`.
The tests use synthesized signals with known payloads passed through a channel
model (carrier offset, phase, AWGN, timing and clock error, band limiting). They
characterize the pipeline; over-the-air performance of the individual Inmarsat
modes will be measured once those demodulators are added.

## Results

| Test | Result |
|------|--------|
| Sensitivity (frame error rate vs SNR) | 0% at 5 dB and above; threshold 4 dB |
| Carrier-offset capture range | 0% error to +/-1.6e-2 cycles/sample |
| Fractional-sample timing (0.5 sample) | 4% |
| Sample-clock error (50 / 200 / 1000 ppm) | 0% / 0% / 0% |
| Band limiting (2 / 3 / 5-tap low-pass) | 44% / 1% / 0% |
| False alarm on pure noise (2000 trials) | 0 |
| Input formats cf32 / cs16 / cu8 | exact / exact / exact |
| Multi-frame stream recovery (8/10/12 dB) | 30/30 each |
| Decode throughput | 17x realtime (Aero channel), 152x (STD-C) |
| Down-converter throughput | 4.8x realtime on 2.048 Msps |

Sensitivity is a clean waterfall down to the AWGN-limited threshold. Carrier
offset, clock error, and fractional timing are handled without a tracking loop by
the preamble-aided frequency and phase estimator. Detection normalizes by window
energy, so no gain calibration is required and no frames are produced from noise.

## Notes and limits

- Heavy band limiting (an effective pulse of two samples or fewer) costs about
  40% of frames. The integrate-and-dump filter is an approximation; the Inmarsat
  modes use root-raised-cosine shaping and a matched filter (`channelize.rrc`).
- Down-conversion filters each input block independently, so a frame that lands
  exactly on a block boundary can be lost. Such a frame fails its CRC and is
  dropped; it is never reported incorrectly. Stateful filtering removes this.
- These figures validate the pipeline, not live reception, which requires a
  receiver and antenna.
