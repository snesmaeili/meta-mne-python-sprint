Draft, following the design discussed in #14206.

`tmin` and `tmax` accept `(n_events,)` arrays. Bounds that carry no actual
variation collapse back to the scalar path, so existing behaviour is unchanged.
`EpochsArray` also takes a list of `(n_channels, n_times_i)` arrays and derives
each epoch's `tmax` from its own length.

New: `durations`, `get_times(epoch)`, `variable_duration`, `as_fixed()`.
`as_fixed()` returns a padded `EpochsArray` together with the number of epochs
contributing at each time point.

`times` raises when durations vary, rather than returning the union. Returning it
would leave `len(epochs.times) == data.shape[-1]` false while looking ordinary.
The union is still available as `as_fixed().times`.

**Method behaviour.** Three groups rather than a blanket fallback:

- native: `pick`, `drop`, `__getitem__`, `shift_time`
- raise: the six reductions (`average`, `standard_error`, `subtract_evoked`,
  `iter_evoked`, `compute_tfr`, `compute_psd`), plus per-trial operations with no
  ragged implementation yet (`filter`, `apply_baseline`, `crop`, `decimate`,
  `resample`, `plot_image`, `plot_topo_image`, `save`, `export`)
- warn and run on `as_fixed()`: `plot`, `to_data_frame`

I originally had the reductions fall back with a warning, as suggested in the
issue. Measuring it changed my mind. On 43 epochs spanning 2.0–3.6 s,
`average()` returns an Evoked that is 44% NaN from the first drop-out onward,
while `nave` reports 43 where 3 epochs remain — one short epoch takes out the
whole time point. `compute_tfr` was worse: it padded and then transformed, which
is the reverse of the order argued for in the issue.

**Tutorial.** `tutorials/epochs/70_variable_duration_epochs.py` builds epochs
straight from the Sleep Physionet hypnogram durations, so no new dataset is
needed and `tools/circleci_download.sh` already prefetches it. Bouts over five
minutes are set aside to keep the padded array small, leaving 130 epochs from
30 to 300 s. It walks through what the object holds, which operations are
unaffected by ragged trials, which ones refuse and why, and closes on the
contributing-count curve from `as_fixed()`: 130 epochs at t=0 falling to 1 by
300 s, which is the reason `average()` cannot return an ordinary `Evoked`. The
last section points at `tut-sleep-stage-classif`, where fixed 30 s windows are
the right representation, so the two are not read as competing.

**Two defects the tutorial found.** Both are fixed here with regression tests
that fail without the fix.

- `load_data()` set `_raw_times` from `self.times`, which refuses once durations
  vary, so `Epochs(raw, ...)` failed outright. Every existing test built through
  `EpochsArray`, so the path from `Raw` had no coverage at all.
- `GetEpochsMixin._item_to_select` returns slices untouched, so `epochs[:10]`
  iterated `np.atleast_1d(slice)`, which yields the slice itself. The data list
  came back holding one nested list and the bounds subsetting then raised on it.
  String and array indexing were unaffected, which is why it stayed hidden.

**What was validated.** Construction and extraction, plus a per-epoch TFR
pipeline built on `get_data()`. On all 24 ds004505 subjects (29,546 swing
cycles), every epoch is byte-identical to the raw slice it came from, and
re-running an existing ERSP analysis through the container reproduces the
previously computed maps at 0.000e+00 dB with matching retained counts. The
tutorial repeats the byte-identity check on annotated sleep recordings. Scripts
and per-subject reports:
https://github.com/snesmaeili/meta-mne-python-sprint/tree/main/validation

This does not validate any padded path — the methods that would need one raise.

`mne/tests/test_epochs.py` passes unchanged; 469 tests pass across epochs,
channels and docstrings.

Open questions are in #14206: whether `tmin`/`tmax` should stay scalar with
per-epoch bounds under separate names, what the reductions should eventually do
about a contributing count that varies over time, and the FIF representation.

AI assistance: I designed the approach and ran the analyses it is validated
against; Claude Opus 5 wrote the implementation and the tests under my
direction, which I reviewed and tested.
