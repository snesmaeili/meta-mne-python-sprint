# Variable-duration epochs in MNE-Python

Working material for MNE sprint board item #10, *"support for variable length
epochs using AwkwardArray?"* — a draft card with no description.

> **Raggedness is a data representation; temporal alignment is an analysis decision.**

This repository contains a design proposal and a runnable prototype. The
prototype needs **no changes to mne-python** — it sits on top of MNE 1.11 and
returns first-class MNE objects.

## Why this, and why now

The feature request has a ten-year paper trail: one WONTFIX (#3533), two open
issues (#5612, #5794), one stalled draft PR (#12315). It has never been
designed, only argued about. Each attempt collapsed the container question into
the alignment question and re-ran the same debate.

We arrived at this from the other end. Our gait and table-tennis pipeline hit the
limitation on real data and had to reimplement epoching outside MNE to get past
it. The production code on the cluster carries the feature request in a comment:

> *"swing cycles differ in duration by construction, so they cannot be stacked
> into one array at all"*
> — `eneuro_merged_ersp.py`, fir, array job 55417405, 24 subjects

## What's here

| | |
|---|---|
| [`docs/04-architecture.md`](docs/04-architecture.md) | The proposal: three layers, the API, and every historical objection answered by name |
| [`docs/05-method-matrix.md`](docs/05-method-matrix.md) | All 61 public `BaseEpochs` methods classified by mathematical meaning |
| [`docs/06-container-benchmark.md`](docs/06-container-benchmark.md) | Does AwkwardArray earn its place? Measured answer |
| `ragged_epochs/` | The prototype: container, ragged-native ops, alignment, ragged TFR |
| `tests/` | 31 tests, including the frequency-preservation invariant |
| `validation/` | End-to-end run on MNE sample data |
| `proposal/` | The board-card description and the issue cross-post |

## Three findings worth leading with

**1. Warping the signal before a time-frequency transform reports the wrong
frequency.** Two trials carrying the same 10 Hz oscillation, at 1.0 s and 2.0 s:

```
native (no warping)          [ 10.0 Hz,  10.0 Hz]
path A  warp signal -> TFR   [ 10.0 Hz,  20.0 Hz]
path B  TFR -> warp TF axis  [ 10.0 Hz,  10.0 Hz]
```

Both give a common time axis; only path B leaves the frequency axis meaningful.
Path A is the mne-mobi prototype; our current production pipeline already does
path B, to a group median anchor, as EEGLAB's `newtimef` has always done. The
correction matters because **path A is what a naive `time_warp_epochs` in MNE
core would ship by default.** Now an executable invariant
(`tests/test_frequency_preservation.py`).

**2. AwkwardArray brings no benefit here.** At MoBI scale (2000 epochs × 128
channels, 0.8–1.7 s), awkward's payload is *byte-identical* to a plain
`list[np.ndarray]` — 587.4 MB both — while being 759× slower on random access
and 12× slower on the jagged reduction it exists to do. Recommendation: a list
plus offsets, zero new dependencies. Details and caveats in
[`docs/06-container-benchmark.md`](docs/06-container-benchmark.md).

**3. Only 6 of 61 `Epochs` methods actually need a common time axis.** 78% need
no cross-trial decision at all; 4 need a stated policy. The "Pandora's box"
objection is real but far smaller than it looks once the API is classified by
mathematical meaning rather than one method at a time.

## Quick look

```python
from ragged_epochs import RaggedEpochs
from ragged_epochs._tfr import compute_tfr, warp_tfr

# mne.Epochs reads annotation onsets but its docstring says the durations
# "are ignored in this case". Read them.
ep = RaggedEpochs.from_annotations(raw, description="stride", context=1.0)
ep.durations          # (n_epochs,) — the experimental data, kept
ep.times              # raises, and names the alignment options

tfr = compute_tfr(ep, freqs)          # ragged time, common frequency axis
tfr.average()                         # raises: time index k isn't the same
                                      # phase of the task in every trial
aligned = warp_tfr(tfr, landmarks, target="median")   # EEGLAB's default
aligned.average()                     # now defined
aligned.to_mne()                      # -> mne.time_frequency.EpochsTFR
```

`context=1.0` carries a second of surrounding recording per epoch so no wavelet
taper straddles a landmark. It is excluded from `durations` and `get_data()`.
Without it, power at the epoch edge — which *is* the cycle-start landmark — comes
out at 0.37 of its true value.

## Run it

```bash
python -m pytest tests/ -q
```

```bash
python validation/end_to_end_sample_data.py
```

```bash
python benchmarks/container_backends.py
```

```bash
python docs/build_method_matrix.py
```

The method matrix is generated against the *installed* MNE, so any method MNE
adds or renames shows up as `UNCLASSIFIED` rather than being silently missed.

Requires MNE ≥ 1.11, NumPy, SciPy. `awkward` is optional and only needed for the
benchmark.

## Status

Prototype and proposal complete; the card is not yet claimed.

V4 — reproducing the poster ERSP on the real recordings — is written but not run
here; the data lives on the cluster. See
`validation/reproduce_poster_ersp.py`, which expresses the completed analysis
through the proposed API and quantifies what the stale signal-domain path would
have cost. Anchor values from the completed run: cohort pooled median contact at
**38.5%** of cycle (per-subject range 32.6–47.9%), cohort median cycle ~1.92 s
against the paper's 1.924 s and 33.3%.
