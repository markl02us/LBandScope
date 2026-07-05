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

## Inmarsat-C (STD-C) receiver

Measured by `python tests/evaluate_stdc.py`. No off-air captures of this signal
are published, so the evaluation reproduces a faithful downlink: the real unique
word, permutation, interleaver, K=7 convolutional code, and scrambler, root-
raised-cosine shaped, then impaired with noise, a carrier offset, and a sample-
clock error. The whole receiver (demodulator, sync, Viterbi, descrambler, parser)
is scored against it. SNR is broadband at 8 samples/symbol; the matched filter
adds about 9 dB of processing gain.

| Test | Result |
|------|--------|
| Sensitivity, frame decode rate, hard-decision Viterbi | 0% error to about -6 dB; collapses by -8 dB |
| Sensitivity, frame decode rate, soft-decision Viterbi | 0% error to about -8 dB; ~-9 dB threshold |
| Carrier-offset tolerance | full rate to +/-400 Hz and beyond |
| Sample-clock tolerance | full rate to 80 ppm |
| End-to-end message integrity (5-message downlink, 0 dB, 180 Hz, 15 ppm) | 5/5 messages, distress flagged, 5/5 positions |
| Decode throughput | ~60x real time |

The dominant sensitivity gain came from carrier recovery, not error correction.
Diagnosis showed the floor was set entirely by frame synchronisation: every frame
that synchronised decoded, and every failure was a sync failure caused by a noisy
residual-frequency estimate rotating the frame. Moving frequency acquisition to a
parabolically-refined squared-spectrum peak fixed synchronisation and lowered the
floor by more than 12 dB. Only then did soft-decision Viterbi contribute its
expected further 2-3 dB — until the sync bottleneck was cleared, soft decisions
made no measurable difference.

### Notes and limits

- Carrier acquisition searches the squared signal within +/-0.8*symbol_rate, so
  it captures offsets up to about +/-0.4*symbol_rate (~480 Hz). Larger offsets
  need the channel roughly centred first, which the application's tuning and
  "Find signal" provide.
- Timing uses maximum-eye-opening sample selection, adequate to 80 ppm — beyond
  any practical receiver clock. A tracking loop would extend this further but is
  not required for real hardware.
- These figures validate the receiver against a faithful signal model.
  Confirmation on a genuine off-air capture is the remaining step.
