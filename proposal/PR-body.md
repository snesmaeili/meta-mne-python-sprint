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

**What was validated.** Construction and extraction, plus a per-epoch TFR
pipeline built on `get_data()`. On all 24 ds004505 subjects (29,546 swing
cycles), every epoch is byte-identical to the raw slice it came from, and
re-running an existing ERSP analysis through the container reproduces the
previously computed maps at 0.000e+00 dB with matching retained counts. Scripts
and per-subject reports:
https://github.com/snesmaeili/meta-mne-python-sprint/tree/main/validation

This does not validate any padded path — the methods that would need one raise.

`mne/tests/test_epochs.py` passes unchanged; 461 tests pass across epochs,
channels and docstrings.

Open questions are in #14206: whether `tmin`/`tmax` should stay scalar with
per-epoch bounds under separate names, what the reductions should eventually do
about a contributing count that varies over time, and the FIF representation.

AI assistance: I designed the approach and ran the analyses it is validated
against; Claude Opus 5 wrote the implementation and the tests under my
direction, which I reviewed and tested.
