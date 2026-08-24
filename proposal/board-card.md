# Support for variable-length epochs

*Proposed description for sprint board #10, item "support for variable length
epochs using AwkwardArray?"*

---

## The framing

> **Raggedness is a data representation; temporal alignment is an analysis decision.**

Three distinct problems have been conflated in every previous discussion:

1. **Container** — how to store trials of unequal duration (#3533, #12315)
2. **Alignment** — how to average processes of unequal duration (#5612)
3. **Per-trial time origin** — rectangular data, different `t=0` (#5794, #11480)

AwkwardArray only addresses (1). Saying so up front is what stops this thread
relitigating #3533 for the fourth time. (3) turns out to be the degenerate case
of (2) where durations happen to be equal, so the same design closes it.

## What this is not

Not "add time warping for gait." Not even "store variable-length epochs using
AwkwardArray." The scope is:

> Introduce a representation and analysis model for variable-duration epochs
> that preserves native trial timing, supports ragged-compatible operations
> unchanged, and requires explicit temporal alignment when a cross-trial
> analysis needs a common coordinate system.

The demand is not gait-specific. #5612's motivating case is mental arithmetic
where the problem-solving phase varies per trial; @drammock's is variable-length
spoken sentences with a target keyword at varying latency. It applies to
decision making, reading, speech, self-paced tasks, sleep segments,
rehabilitation, and MoBI.

## Prototype

A working prototype requiring **no changes to mne-python** is at
`snesmaeili/meta-mne-python-sprint`. 30 tests; end-to-end run on
`mne.datasets.sample`; returns first-class `mne.time_frequency.EpochsTFR`.

### Three layers

```
LAYER 1  container          true durations; no warping, padding, or fake sfreq
LAYER 2  ragged-native ops  per-trial; no kernel rewrites
LAYER 3  explicit alignment crop / pad / normalise / landmark warp / TFR warp
```

### Layer 1 — the cheapest possible first PR

`mne.Epochs` has accepted `events=None → raw.annotations.onset` since 1.7
(#12311), and the docstring says outright:

> "the durations of the annotations are ignored in this case"

`Annotations` already carry per-event `duration`. **The ragged lengths are
already in the data model and are being deliberately discarded.** Reading them
is smaller and more reviewable than any new constructor signature — and it is
why #12315 was blocked on #12311 landing.

`epochs.times` **raises** on ragged data rather than returning the shortest
common interval. A plausible-but-wrong time axis is the easiest way for this
feature to produce quiet scientific errors. The error names `durations`,
`get_times()` and `align_time()`.

`info["sfreq"]` stays the physical sampling rate. Our own prior code set
`info["sfreq"] = n_points - 1` to fake a 0–100% axis; after that, "600 ms" and
"60% of trial" were the same number in the same field. Normalisation belongs in
provenance.

### Layer 2 — most of the API already works

`BaseEpochs` has 61 public methods. Classified by mathematical meaning rather
than one at a time:

| Class | N | |
|---|---:|---|
| Bookkeeping | 22 | no time axis involved |
| Spatial | 12 | channel axis only |
| Naturally ragged | 10 | per-trial; maps over epochs |
| Length-changing | 4 | crop, decimate, resample, shift_time |
| **Ragged with declared policy** | **4** | PSD + the plot_psd family |
| **Ragged output** | **1** | `compute_tfr(average=False)` |
| **Requires a common time axis** | **6** | `average`, `standard_error`, `iter_evoked`, `plot_image`, `plot_topo_image`, `subtract_evoked` |
| IO | 2 | `save`, `export` |

**48 of 61 (78%) need no cross-trial decision at all.** Every layer-2 operation
is asserted equal to stock MNE run per-epoch, so "no kernel rewrites" is
checkable rather than asserted. Full table with per-method notes in the repo.

The policy row is where the science is. Fixed-length epochs silently enforce
`sample weighting == epoch weighting`; ragged trials break that identity and
force it into the open. `ICA.fit` already concatenates to `(n_channels,
n_epochs × n_times)`, so ragged ICA is natural — but should a 5 s trial carry
five times the influence of a 1 s trial? Same question for covariance, and it
propagates into the inverse operator's noise model. **Ragged data exposes
assumptions that fixed-length epochs currently hide.**

### Layer 3 — and the finding that matters most

EEGLAB's `newtimef` warps **after** the time-frequency transform (magnitude by
warp-matrix multiply, phase circularly via `angtimewarp`), to the **median**
landmark latencies. Warping the signal first rescales the time axis the
oscillations live on. Measured, two trials with the same 10 Hz oscillation at
1.0 s and 2.0 s:

```
native (no warping)          [ 10.0 Hz,  10.0 Hz]
path A  warp signal -> TFR   [ 10.0 Hz,  20.0 Hz]
path B  TFR -> warp TF axis  [ 10.0 Hz,  10.0 Hz]
```

Both produce a common axis; only path B reports the right frequency. This is an
executable invariant in the prototype, not a docstring warning. Signal-domain
warping remains available — it is correct for ERP-shaped questions — but the
result carries `alignment.warps_spectral_content == True`.

DTW is deliberately out of v1: unlike biomechanical landmarks it derives
correspondence from signal similarity, so it can align noise and change apparent
component durations.

## On AwkwardArray specifically

**Measured, and it does not earn its place.** 2000 epochs × 128 channels @
250 Hz, durations 0.8–1.7 s:

| backend | payload | random access | jagged reduce |
|---|---:|---:|---:|
| `list[np.ndarray]` + offsets | 587.4 MB | 1× | 1× |
| padded + lengths | 870.4 MB | 5.0× slower | 1.7× slower |
| `awkward.Array` | **587.4 MB** | **759× slower** | **12× slower** |

Awkward's payload is byte-identical to a plain list — both store exactly the
true samples. It saves memory relative to *padding*, but so does a list, without
a new hard dependency. The `reduce` column is the per-epoch mean, i.e. exactly
the jagged reduction awkward exists for; all three implementations agree to
1e-16.

Two structural points for anyone who wants to revisit this:

- The only awkward type that enforces "only time is ragged" is
  `n_epochs * var * n_channels * float64`. The intuitive
  `ak.Array([b.T for b in blocks])` yields `n_epochs * var * var` — the channel
  dimension goes ragged too, silently, with no error.
- That correct layout is `(epoch, time, channel)`, while MNE is channel-major
  everywhere, so every dense conversion pays a transpose.

**This does not close the door.** If a use case has genuinely nested raggedness
— ragged channels *and* ragged time, as in #11705/#12219 — the calculus changes.
The finding is scoped to one ragged axis at EEG scale.

## Answering the objections

| | Objection | Answer |
|---|---|---|
| **O1** | @agramfort (#12315): padding makes `nave` a function of time, and `nave` scales the noise covariance in the inverse. *This, not the SciPy wall, is what stopped #12315.* | No automatic reduction over ragged data — `average()` raises. Aligned by warping, `nave` is constant by construction. Aligned by padding, `pad()` **returns the `nave(t)` vector** instead of hiding it behind a scalar. |
| **O2** | @kingjr (#3533): "just use rERP" — modern form, deconvolution (`unfold`). | Complementary. Deconvolution solves overlap and continuous covariates; it does not give the ERSP of a variable-duration process aligned to its own internal landmarks. |
| **O3** | @alexrockhill (#12315): SciPy ignores `np.ma` masks. | Still true — verified on SciPy 1.17.1, `spectrogram` returns a plain `ndarray` and treats padded samples as real data. Answered architecturally: ragged for storage/indexing/IO, dense at the point of computation. Applies identically to awkward, so it is not an argument for awkward over masks. |
| **O4** | @mmagnuski (#3533): FieldTrip's variable-length trials "were never that useful in practice". | Fair and instructive. FieldTrip stores ragged (cell arrays) but never built the alignment layer, so users hit the same wall one level later. An argument for layers 2–3, not against layer 1. |
| **O5** | mne-connectivity #142 shipped padding. | Correct for that problem. Cited as precedent for "padding is fine *at the computation boundary*" — this design's position. |
| **O6** | @kingjr (#3533): Epochs are by definition comparable repetitions. | The variable-duration process *is* the repetition. Cutting #5612's trials to a common `tmax` discards the part being studied. |

## Scope for the sprint

Layer 1 only: representation, indexing, per-epoch times, the backend benchmark,
and the method matrix. Layers 2–3 exist in the prototype as evidence that the
design closes, but land as later PRs.

@larsoner asked for a dev meeting on this in 2023 and never got one. The sprint
is that meeting; this is the material to hold it against.

## Prior art and references

- Gwin, Gramann, Makeig & Ferris (2011), *NeuroImage* 54:1289–1296 — canonical gait-locked ERSP
- Studnicki & Ferris (2023), *eNeuro* 10(4) ENEURO.0463-22.2023
- Ehinger & Dimigen (2019), *PeerJ* 7:e7838 — `unfold`, the deconvolution alternative
- Landmark registration in functional data analysis: Kneip & Gasser 1992; Ramsay & Li 1998; Ramsay & Silverman 2005. Implemented in Python by `scikit-fda` (BSD) — evaluate as a dependency rather than reimplementing.
- EEGLAB `newtimef.m` / `timefreq.m`, `timestretch` option

**Known weakness, disclosed rather than hidden:** averaging time-warped ERSPs
across subjects with differing gait characteristics can shift or reverse the
phase of power changes relative to single-subject results. Warping is not free.
