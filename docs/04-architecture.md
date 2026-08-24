# Architecture

> **Raggedness is a data representation; temporal alignment is an analysis decision.**

Every previous discussion of this feature collapsed those two things. #3533
argued about containers and got answered with "use rERP" — an alignment answer.
#5612 says explicitly it is *not* about the container and gets pointed at #3533
anyway. #12315 started as a PSD entry point and turned into a container debate.

Separating them is the whole design.

```
Raw + per-trial bounds
        │
        ▼
┌───────────────────────────┐
│ LAYER 1  RaggedEpochs     │   true durations, real seconds
│ no warping, no padding,   │   info["sfreq"] stays physical
│ no fake sfreq             │
└───────────┬───────────────┘
            │
   ┌────────┴────────┐
   │                 │
   ▼                 ▼
LAYER 2           LAYER 3
ragged-native     explicit alignment
operations        (never automatic)
   │                 │
filter, baseline   ┌─┴──────────────────┐
detrend, Hilbert   │ common crop        │
reference, TFR     │ pad (+ nave(t))    │
PSD, ICA, cov      │ duration-normalise │
   │               │ landmark warp      │
   │               │ TFR-domain warp    │
   │               └─┬──────────────────┘
   │                 │
   └────────┬────────┘
            ▼
   common-grid analyses
   average, ITC, stats, decoding
```

Issues #5794 and #11480 (per-trial time origin on *rectangular* data) are the
degenerate case where durations happen to be equal. `RaggedEpochs` with uniform
lengths is bit-identical to `mne.EpochsArray` — asserted in
`tests/test_parity_with_mne.py::test_uniform_ragged_epochs_equal_stock_epochs`.
Those issues get closed by the same design rather than forgotten.

---

## Layer 1 — the container

```python
epochs.durations        # (n_epochs,) seconds — the experimental data
epochs.lengths          # (n_epochs,) samples
epochs.tmin, epochs.tmax  # (n_epochs,) each
epochs.is_uniform       # bool
epochs.get_times(i)     # one epoch's time vector
epochs.get_times()      # all of them, ragged
epochs.get_data(representation='ragged'|'dense'|'concatenated')
epochs.times            # RAISES unless a common axis genuinely exists
```

### `times` raises. This is the highest-risk API decision in the design.

Returning the shortest common interval, or the first epoch's axis, would make
most code "work". It would also make it quietly wrong — the single easiest way
for this feature to produce scientific errors that never surface. So:

```
RaggedTimesError: These 24 epochs do not share a time axis
(durations 0.851-1.349 s).
There is no single correct answer here, so pick one explicitly:
  .durations            per-epoch duration in seconds
  .get_times(i)         the time vector of one epoch
  .get_times()          all time vectors, ragged
  .align_time(...)      produce a common axis (crop / pad /
                        normalise / landmark warp), then .times
```

### `sfreq` stays physical

mne-mobi's `time_warp_epochs` set `info["sfreq"] = n_points - 1` to fake a
0–100% axis. After that, "600 ms" and "60% of trial" were the same number in the
same field, and the original durations were unrecoverable. Here the sampling
frequency is always a real rate over a real span, and normalisation is recorded
in provenance instead. Pinned by
`tests/test_landmark_alignment.py::test_no_fake_sfreq`.

### Entry point — the cheapest first PR that exists

`mne.Epochs` has accepted `events=None → raw.annotations.onset` since 1.7
(PR #12311), and its docstring says outright:

> "the durations of the annotations are ignored in this case"

`Annotations` already carry per-event `duration`. **The ragged lengths are
already in the data model and are being deliberately discarded.** Reading them
is a smaller, more reviewable change than any new constructor signature — and it
is why #12315 was blocked on #12311 landing in the first place.

`RaggedEpochs.from_annotations(raw, description='stride')` does exactly this.

Do **not** propose `mne.Epochs(raw, events, tmin=-0.2, tmax=[1.24, 2.17, ...])`.
A per-epoch array silently changes the return type and duplicates information
already available.

### Storage backend: `list[np.ndarray]` + offsets

Settled by measurement, not assumption — see
[06-container-benchmark.md](06-container-benchmark.md). Awkward gives *identical*
memory to a plain list (587.4 MB both) while being 759× slower on random access
and 12× slower on the jagged reduction it exists for. Three backends are
implemented behind one interface so the result is reproducible and so the
decision can be revisited if a use case with genuinely nested raggedness
(#11705/#12219, ragged *channels*) arrives.

### The ndarray boundary

**Ragged for storage, indexing, metadata and IO; dense at the point of
computation, one epoch at a time.**

This is forced, not chosen. On SciPy 1.17.1 today, `scipy.signal.spectrogram`
given a `np.ma.MaskedArray` returns a plain `ndarray` with the mask silently
dropped and padded samples treated as real data. That was alexrockhill's blocker
on #12315 in 2023 and it is still live. It applies identically to awkward, so it
is not an argument for one ragged container over another — it is an argument for
never handing a ragged object to a numerical kernel.

---

## Layer 2 — ragged-native operations

Anything mathematically per-trial maps over epochs with **no kernel rewrite**.
mne-mobi's `compute_single_ersp` already proved this: it builds a one-trial
`Epochs` at that cycle's own duration and runs an ordinary Morlet TFR.

Every layer-2 operation is asserted equal to stock MNE run per-epoch
(`tests/test_parity_with_mne.py`). That is what makes "no kernel rewrites" a
checkable claim rather than a hope.

**The honest cost:** this is a Python-level loop with per-call overhead, and it
is exactly why `compute_single_ersp` is slow. The framing is deliberate —
map-over-epochs is the *correctness reference and parity oracle*; vectorised
fast paths get added where padding overhead is acceptable, never the other way
round.

### The policy row is where the science is

Fixed-length epochs silently enforce `sample weighting == epoch weighting`.
Ragged trials break that identity and force it into the open:

```python
concatenate_for_decomposition(epochs, weighting='samples')  # long trials dominate
concatenate_for_decomposition(epochs, weighting='equal')    # each trial counts once
```

- **ICA** — `ICA.fit` already reshapes `Epochs` to `(n_channels, n_epochs ×
  n_times)`, so nothing about ICA requires equal lengths. But should a 5 s trial
  carry five times the influence of a 1 s trial? `'samples'` is right when
  modelling the data-generating process; `'equal'` is right when trials are
  experimental units and duration is a nuisance variable.
- **Covariance** — same question, and it propagates into the inverse operator's
  noise model.
- **PSD** — the policy is the frequency grid. Fixing `n_fft` for all epochs caps
  resolution at the shortest one. Stating that beats letting each epoch return a
  different `freqs` and interpolating later.

There is no defensible default, so these functions return the weights alongside
the data instead of quietly picking one.

**This is a feature of the proposal, not a caveat: ragged data exposes
assumptions that fixed-length epochs currently hide.**

---

## Layer 3 — explicit alignment

Never automatic. `average()` does not call it; `average()` raises and points
here.

| Strategy | Meaning | Use |
|---|---|---|
| none | preserve real time and duration | single-trial analyses |
| `common-crop` | keep only the interval present in every trial | ERP, simple stats |
| `pad` | keep all samples, mark unavailable time | storage, interchange |
| `duration-normalise` | start/end → common phase | simple cyclic behaviour |
| `landmark` | internal events aligned | gait, reaching, RT tasks |
| **TFR-domain warp** | alignment without warping spectral frequencies | **ERSP — the default for TF work** |
| DTW | nonlinear, signal-driven | **deferred out of v1** |

DTW is deferred deliberately. Unlike biomechanical landmarks it derives
correspondence from signal similarity, so it can align noise and change apparent
component durations. It belongs as a later pluggable registration algorithm.
RIDE belongs in background: it solves latent-component latency variability, a
different problem from variable-duration storage.

### C1 — warp the TFR, not the signal

This is the correction that matters most, and it is now an executable test
rather than a docstring warning.

EEGLAB's `newtimef` warps **after** the time-frequency transform. From
`timefreq.m`:

```matlab
timerefs = median(g.timestretch{1}', 2);
M   = timewarp(marksPos, refsPos);
TSr = transpose(M * r');        % magnitude, by warp-matrix multiply
TStheta = angtimewarp(...);     % phase, circularly
TStmpall = TSr .* exp(i * TStheta);
```

mne-mobi's `channel_tfr_general.warp_sig` warps the **raw signal** first, then
runs `tfr_multitaper`. Stretching a trial before the transform rescales the time
axis its oscillations live on.

Measured, two trials carrying the same 10 Hz oscillation at 1.0 s and 2.0 s:

```
native (no warping)          [ 10.0 Hz,  10.0 Hz]
path A  warp signal -> TFR   [ 10.0 Hz,  20.0 Hz]   <- mne-mobi warp_sig
path B  TFR -> warp TF axis  [ 10.0 Hz,  10.0 Hz]   <- EEGLAB newtimef
```

Both give a common axis. Only path B reports the right frequency.
(`tests/test_frequency_preservation.py`.)

Signal-domain warping is still reachable — it is correct for ERP-shaped
questions — but the resulting object carries
`alignment.warps_spectral_content == True` and says so in its summary.

**Phase must be warped circularly.** Linear interpolation between +179° and
−179° gives 0°, not 180°. EEGLAB uses `angtimewarp`; `_warp_complex`
interpolates the unit complex vector and takes its argument, which is the same
idea without unwrapping ambiguities.

### C2 — warp to the median, not to evenly-spaced positions

EEGLAB defaults to the **median** landmark latencies across trials
(`timerefs = median(...)`), which is what the gait-EEG literature means by
"linearly time-warped to the group median gait cycle length".

mne-mobi's `warp_sig` used `rel_pos = np.linspace(0, 1, n_anchors)` — evenly
spaced. With the five canonical gait anchors that forces:

```
real gait:   RHS ---- LTO -------------- LHS ---- RTO -------------- RHS
              0%      12%                50%      62%               100%

'uniform':   RHS ------- LTO ------- LHS ------- RTO ------- RHS
              0%         25%         50%         75%        100%
```

Stance is ~60% of the gait cycle and swing ~40%; `uniform` reports 50/50 and
puts toe-off at 25% instead of 12%. This is a straight bug against the reference
implementation. `target='uniform'` still exists, named so the divergence is
never silent. Pinned by
`test_median_target_preserves_stance_swing_proportions`.

### Mismatched landmark counts are refused

mne-mobi's `channel_tfr_general` accepted 2–5 anchors per cycle and mapped
whatever it found onto evenly-spaced positions — so a cycle missing LHS had its
RTO silently warped onto another cycle's LHS. `keep_only_five_anchor` existed as
a workaround. Refusing with an explanation is better than a flag:

```
All epochs must have the same number of landmarks. Got 4 anchors: 7 epochs;
5 anchors: 15 epochs.
Mapping 4 anchors and 5 anchors onto one target silently aligns different
biomechanical events: a cycle missing LHS would have its RTO warped onto
another cycle's LHS.
Either drop the incomplete cycles, or align each anchor-count group separately
and compare them explicitly.
```

### Landmarks come from `metadata` — reuse, don't rebuild

`Epochs.add_annotations_to_metadata()` already produces `annot_onset` /
`annot_duration` / `annot_description` list-columns, with onsets **already
relative to epoch t=0**. `epochs.get_annotations_per_epoch()` gives the same
thing as tuples. That is exactly the input a landmark spec needs, so:

```python
landmark_warp(epochs, 'annot_onset', target='median')
```

No new per-epoch metadata mechanism is required.

---

## Provenance is first-class

```python
AlignmentRecord(
    method='piecewise-linear', domain='tfr', target_coord='seconds',
    interpolation='linear',
    original_duration=..., original_start=..., original_end=...,
    original_landmarks=[...], target_landmarks=[...],
    target_rule='median', landmark_names=('RHS','LTO','LHS','RTO','RHS_next'),
)
```

`epochs.durations` still returns the experimental durations after alignment.
`alignment.warps_spectral_content` tells a reader whether the frequency axis can
be trusted. A user can never confuse 600 ms with 60% of a trial.

mne-mobi could not do this because normalised phase had to be forced back
through `Epochs`' single `sfreq`/`times` model.

---

## Answering the objections by name

| # | Objection | Answer |
|---|---|---|
| **O1** | **agramfort, #12315** — padding makes the noise level time-dependent; `nave` becomes a function of time; `nave` scales the noise covariance in the inverse. *This, not the SciPy wall, is what stopped #12315.* | No automatic reduction over ragged data: `average()` raises. Aligned by warping, `nave` is constant by construction. Aligned by padding, `pad()` **returns the `nave(t)` vector** rather than hiding it behind a scalar. Pinned by `test_pad_returns_time_dependent_nave`. |
| **O2** | **kingjr, #3533** — "just use rERP"; modern form: deconvolution (`unfold`, Ehinger & Dimigen 2019). | Complementary, not competing. Deconvolution solves overlap and continuous covariates. It does not give the ERSP of a variable-duration process aligned to its own internal landmarks. Say which question goes to which; do not claim superiority. |
| **O3** | **alexrockhill, #12315** — SciPy ignores `np.ma` masks. | Confirmed still true on SciPy 1.17.1. Answered architecturally: dense at the computation boundary. Applies identically to awkward, so it is not an argument for awkward over masks. |
| **O4** | **mmagnuski, #3533** — FieldTrip's variable-length trials "were never that useful in practice". | Fair, and instructive. FieldTrip stores ragged (`data.trial`/`data.time` cell arrays) but never built the alignment layer, so users hit the same wall one level later. An argument for layers 2–3, not against layer 1. |
| **O5** | **mne-connectivity #142** — the ecosystem's one shipped ragged decision chose padding. | Correct for that problem (ragged seeds/targets, no time axis). Cited as precedent for "padding is fine *at the computation boundary*", which is this design's position. |
| **O6** | **kingjr, #3533** — Epochs are by definition comparable repetitions. | The variable-duration process *is* the repetition. #5612's mental-arithmetic trials are comparable cognitive events of unequal duration; cutting them to a common `tmax` discards the part being studied. |
