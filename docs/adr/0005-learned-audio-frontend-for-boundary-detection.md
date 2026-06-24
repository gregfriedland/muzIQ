# ADR 0005: Learned Audio Frontend For Boundary Detection

## Status

Proposed. The next frontend experiment should use a LEAF-style learned
filterbank as the primary replacement for the current fixed log-frequency FFT
fold, while retaining explicit onset-salience features as diagnostic and timing
support.

## Date

2026-06-24

## Context

The current boundary model uses fixed audio features from
`SourceTrackingAudioConfigV2`:

- sample rate: 16 kHz;
- hop: 80 samples, or 5 ms;
- FFT window: 512 samples, or 32 ms;
- log-frequency bands: 40 bands from 40 Hz to 8 kHz.

The model input bands are log-spaced, but they are produced by pooling linear
FFT bins. This creates two limitations:

1. Forty log bands are too coarse for musical pitch and timbre. Across 40 Hz to
   8 kHz, 40 bands cover roughly 2.3 semitones per band.
2. Increasing the number of log bands without changing the frontend is limited
   by the linear FFT bin spacing underneath. A 512-sample FFT at 16 kHz has
   31.25 Hz bin spacing. Fine low-frequency log bands can collapse onto one or
   a few FFT bins, becoming redundant or noisy.

Recent phase 2 work showed that source count is not the main bottleneck. The
model can keep the active source count stable, but onset ranking still fails on
same-instrument re-attacks and post-onset shoulders. Better frontend features
should therefore improve two things:

- represent timbre and pitch shape more finely than 40 fixed bands;
- preserve sharp onset evidence without adding excessive latency.

## Options Considered

### VQT

Variable-Q transform (VQT) is a deterministic time-frequency transform with
approximately log-frequency bins. It is similar to a constant-Q transform, but
with more flexible bandwidths.

Advantages:

- gives musically meaningful log-frequency bins directly;
- is deterministic and easy to inspect;
- can be configured with semitone, half-semitone, or finer resolution;
- is a strong baseline for replacing the current fixed FFT fold.

Disadvantages:

- low-frequency bins use longer windows, which can smear bass and organ onsets;
- compute can be higher than the current FFT fold;
- it cannot adapt its frequency allocation to the actual onset-ranking loss;
- it still needs explicit onset salience or spectral flux features for timing.

VQT is the best deterministic baseline, but it is not the best long-term
frontend if the model needs to learn which frequency regions and bandwidths are
most useful for re-attack detection.

### SincNet

SincNet uses parameterized band-pass filters on raw waveform. Each filter learns
only its low and high cutoff frequencies.

Advantages:

- extremely parameter efficient;
- interpretable, because every filter remains a band-pass filter;
- lower latency is possible if filter lengths are kept short;
- avoids hard-coding the current 40-band spacing.

Disadvantages:

- less expressive than LEAF because each filter is constrained to a simple
  band-pass shape;
- needs a separate envelope, compression, and normalization stage;
- may be brittle if cutoff frequencies drift or collapse without strong
  regularization;
- is less directly aligned with the existing log-spectrogram-style model input.

SincNet is attractive if interpretability and parameter count dominate, but it
does not give enough control over pooling and compression by itself.

### LEAF

LEAF is a learned audio frontend that replaces fixed mel or log filterbanks
with learnable filters, pooling, compression, and normalization. In this ADR,
"LEAF-style" means a constrained learned filterbank initialized near a sensible
audio representation, not an unconstrained raw-waveform CNN.

Advantages:

- can learn frequency centers and bandwidths instead of fixing them to 40 log
  bands;
- can keep a log-frequency prior while adapting to the actual boundary loss;
- can learn compression and normalization that reduce loudness and noise
  sensitivity;
- is small compared with pretrained speech encoders such as wav2vec or HuBERT;
- can produce the same kind of time-varying frame embedding expected by the
  existing encoder;
- can be constrained to a target output hop, such as 10 ms, to control latency.

Disadvantages:

- harder to debug than VQT because the frontend changes during training;
- needs careful initialization and regularization;
- pooling windows can blur onsets if allowed to grow too large;
- requires frontend-specific metrics to ensure learned filters do not collapse.

LEAF is the best fit because the current problem is not just "more bins." It is
learning a representation that separates true re-attacks from sustained energy,
post-onset shoulders, and timbral drift. LEAF gives the model enough freedom to
adapt frequency resolution, bandwidth, compression, and pooling while remaining
structured enough to avoid the data hunger of a fully unconstrained raw CNN.

## Decision

Use a LEAF-style learned frontend as the preferred next frontend experiment.

The initial target should be:

- raw waveform input at 16 kHz;
- 96 to 192 learned filters, initialized on a log or mel-like scale;
- output hop of 10 ms;
- bounded pooling widths to limit onset smear;
- learned compression or PCEN-style normalization;
- concatenate or preserve explicit onset-salience channels during the first
  experiments.

The current FFT-fold frontend should remain available as the control. VQT should
be implemented as a deterministic comparison baseline if LEAF is ambiguous.
SincNet should remain a lower-priority alternative for an interpretable learned
filterbank.

## Consequences

LEAF changes the frontend from fixed signal processing to a trainable component.
That means improvements can come from the representation itself, but failures
can also come from frontend instability. The experiment must therefore report
frontend-specific diagnostics, not only onset AP:

- learned center frequencies and bandwidths;
- effective pooling widths;
- per-filter activation statistics;
- onset sequence AP and best-threshold metrics;
- false-positive and false-negative breakdowns by instrument family and timing
  bucket.

Latency must be treated as a first-class metric. A 10 ms output hop is a better
fit than the current 5 ms hop for this task because it reduces redundant frames
while keeping onset timing well below the current 25 ms shoulder tolerance.
Pooling and filter lengths must be bounded so the frontend does not hide
re-attacks under long smoothing windows.

## Follow-Up

Implement this as a staged frontend comparison:

1. Add a pluggable frontend setting while preserving the current FFT-fold path.
2. Add a LEAF-style frontend that emits frame embeddings compatible with the
   existing encoder.
3. Train phase 1 on notes480 with LEAF and early stopping on onset sequence AP.
4. If phase 1 improves or matches the FFT-fold baseline, run phase 2 from the
   LEAF phase 1 checkpoint.
5. If LEAF is unstable, run a VQT baseline with roughly 96 to 128 log-frequency
   bins and a 10 ms hop to separate "more musical frequency resolution" from
   "learned frontend" effects.

