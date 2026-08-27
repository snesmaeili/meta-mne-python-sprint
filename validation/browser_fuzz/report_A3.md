# A3 — the selection and drop lifecycle

Harness added: `a3_lifecycle.py` (tagged-epoch factory + bounds/length auditor),
`a3_getitem.py`, `a3_browser.py`, `a3_reorder.py`, `a3_ops.py`, `a3_method_sweep.py`,
`a3_fixed_control.py`, `a3_fix_probe.py`, `a3_repro.py`.

Every epoch carries a flat tag on channel 0 naming its **source** index, so after any
reorder/subset/drop the auditor recovers which source epoch sits in each slot *from the
data itself* and checks `_tmin_per_epoch`/`_tmax_per_epoch`/`events`/`selection`/`_data`/
`metadata` against it. No expectation is read from the object under test.

```bash
QT_QPA_PLATFORM=offscreen QT_QPA_FONTDIR="C:/Windows/Fonts" MPLBACKEND=Agg \
PYTHONPATH="D:/mne-python;D:/meta-mne-python-sprint;D:/meta-mne-python-sprint/validation/browser_fuzz" \
python a3_repro.py
```

**Headline: the parallel-array invariant holds everywhere it could be pushed.**
`_tmin_per_epoch`/`_tmax_per_epoch` never drifted from `events`/`selection`/`_data` across
450 subselections × 2 backends, 16 index forms, 20 mark-and-close scenarios, and 126
`shift_time` configurations. The three defects below are the *other* half of the
bookkeeping: the union time axis `_getitem` forgets to re-derive, and two lifecycle
methods the frozen classification has no row for.

---

## A3-1 — subselecting or dropping leaves the shared time axis at the *parent's* span, so `as_fixed()` invents padding for epochs that are gone

**Verdict: new defect, ragged-only. Silent wrong data.** Fixed path clean on branch and at
base. Independently re-verified by the coordinator.

`GetEpochsMixin._getitem` ([mne/utils/mixin.py:275](D:/mne-python/mne/utils/mixin.py:275))
moves the per-epoch bounds with the epochs — correctly — but never re-derives
`_raw_times`, which `__init__` built as *the union of those bounds*
([mne/epochs.py:834](D:/mne-python/mne/epochs.py:834)) and which `as_fixed()` reads back
([mne/epochs.py:1103](D:/mne-python/mne/epochs.py:1103)).

**`crop()` does re-derive it**, in the same file 2500 lines later
([mne/epochs.py:2841](D:/mne-python/mne/epochs.py:2841)). The two subsetting paths
disagree.

Four epochs of 100/250/75/180 at 100 Hz, `tmin = [0, -0.2, 0.1, -0.5]`:

| | value | should be |
|---|---|---|
| `epochs[0]` holds | 1 epoch, 100 samples, `[0.00, 0.99]` | — |
| `_raw_times` after | `[-0.50, +2.29]`, **280 samples** | `[0.00, 0.99]`, 100 |
| `as_fixed()` | `(1, 3, 280)`, **540 of 840 values NaN** | `(1, 3, 100)`, 0 NaN |
| `n_contributing == 0` at | **180 of 280 time points** | 0 |
| `to_data_frame()` | **280 rows** | 100 |

`n_contributing == 0` is the sharp end: the PR returns that array precisely so a reader can
see how many epochs back each time point, and here it reports 180 time points backed by
**nothing at all**, because they belong to epochs that were dropped.

**The fallback warning contradicts its own result within one call.**
`_wrap_variable_fallback` builds its message from `self.tmin.min()`/`self.tmax.max()`
([mne/epochs.py:4221](D:/mne-python/mne/epochs.py:4221)), which *are* updated:

> `to_data_frame() … ran on as_fixed(): every epoch padded to span 0 to 0.99 s.`

…while `as_fixed()` actually padded to −0.5 → 2.29 s. The one diagnostic pointing at the
padding names the right window while the data uses the wrong one.

**The browser's own close path reaches it.** Browse 4 ragged epochs, mark 3 bad, close —
`_close_impl` → `inst.drop(bad_ixs)` → `_getitem`. One 100-sample epoch survives and
`to_data_frame()` returns 280 rows. Same via `equalize_event_counts(method="mintime")`.

The tutorial states this is safe — *"Selecting epochs, selecting channels and dropping
epochs all work as usual … The per-epoch bounds travel with the epochs they belong to"*
(`tutorials/epochs/70_variable_duration_epochs.py:172`). The bounds do travel. The axis
they define does not.

### A3-1b — a subset can never be recognised as fixed-duration again

`crop()` re-collapses `_variable_duration` when the variation is gone; `_getitem` doesn't:

```python
ep = make((100, 100, 250))   # all tmin 0
two = ep[[0, 1]]             # two 100-sample epochs, one shared axis
two._variable_duration       # -> True
two.times                    # -> RuntimeError
two.average()                # -> NotImplementedError
```

and the message refutes itself:

> `These 2 epochs have durations from 0.990 to 0.990 s, so there is no time axis they share.`

`ep.crop(0.0, 0.99)` on the same object gives `_variable_duration == False`.

### Fix — verified by monkeypatch

At [mne/utils/mixin.py:275](D:/mne-python/mne/utils/mixin.py:275), after the two bound
slices:

```python
if getattr(inst, "_variable_duration", False):
    inst._tmin_per_epoch = inst._tmin_per_epoch[select]
    inst._tmax_per_epoch = inst._tmax_per_epoch[select]
    if len(inst._tmin_per_epoch):          # epochs[[]] must stay legal
        sfreq = float(inst.info["sfreq"])
        start_idx = int(round(inst._tmin_per_epoch.min() * sfreq))
        stop_idx = int(round(inst._tmax_per_epoch.max() * sfreq))
        inst._raw_times = np.arange(start_idx, stop_idx + 1) / sfreq
        inst._set_times(inst._raw_times)
```

Measured with the patch: `[0]` → 100 samples, `[0,2]` → 100, `[1]` → 250, `[:]` → 280
(unchanged), `n_contributing == 0` nowhere, `to_data_frame()` 100 rows. Fixed path
byte-identical; `epochs[[]]` does not raise; browser boundaries unchanged on both backends.

For A3-1b, the sample-index/length comparison at `mne/epochs.py:2846` is the block to
reuse — but whether `_getitem` should re-collapse is a **decision**, not an oversight to
patch silently. It simply should not differ from `crop`.

---

## A3-2 — `drop_bad(reject=…)` / `drop_bad(flat=…)` raise an internal `RuntimeError`

**Verdict: new defect, ragged-only.** Fixed path native on branch and at base.

```python
ep.drop_bad(reject=dict(eeg=200e-6))
# RuntimeError: These 5 epochs have durations from 0.740 to 2.490 s,
# so there is no time axis they share. …
```

Path: `drop_bad` → `_get_data(out=False)` ([mne/epochs.py:1982](D:/mne-python/mne/epochs.py:1982))
→ `_handle_tmin_tmax` ([mne/utils/mixin.py:601](D:/mne-python/mne/utils/mixin.py:601),
`n_times = self.times.size`) → `Epochs.times` raises.

`drop_bad` is on **none** of `_VARIABLE_FALLBACK` / `_VARIABLE_NEEDS_POLICY` /
`_VARIABLE_NOT_IMPLEMENTED`, so the user gets an internal error rather than one of the
PR's three deliberate messages.

**The ragged implementation already exists and is unreachable.**
`_load_variable_from_raw` ([mne/epochs.py:1354](D:/mne-python/mne/epochs.py:1354)) does
exactly this rejection per epoch via `_is_good_epoch(epoch, n_times=self._n_times_per_epoch(idx))`,
updating bounds/events/selection/metadata together. Its only caller is `load_data()` under
`if not self.preload`, and `_check_variable_unsupported` forbids non-preloaded ragged
epochs. The identical rejection at construction works:

```python
mne.Epochs(raw, ev, tmin=array, tmax=array, preload=True, reject=dict(eeg=100e-6))
# kept 4 of 5, selection [0, 2, 3, 4], drop_log ((), ('b',), (), (), ())
```

Rejection is a per-trial peak-to-peak reduction, well defined for ragged epochs — the same
argument as F3's histogram. Routing `drop_bad`'s preloaded branch through the existing
`_n_times_per_epoch` check is the expected fix. A blanket `_VARIABLE_NOT_IMPLEMENTED` row
would also break the **no-arg** `drop_bad()`, which currently short-circuits and which
`_concatenate_epochs` calls internally (`mne/epochs.py:5540`, `:5598`).

---

## A3-3 — `concatenate_epochs` raises the same internal error, with a count naming the wrong epochs

**Verdict: new defect, ragged-only.** Fixed path native on branch and at base.

Fails at [mne/epochs.py:5562](D:/mne-python/mne/epochs.py:5562),
`if not np.allclose(epochs.times, epochs_list[0].times)`. The message reports on the second
list element only, so a user joining 4 epochs is told about 2.

**This is the classification-list gap, and it is structural**: `concatenate_epochs` is a
module-level function, so the three `setattr` loops at
[mne/epochs.py:4301](D:/mne-python/mne/epochs.py:4301) cannot reach it however the dicts
are edited. Concatenation is mathematically trivial for ragged epochs (append the lists,
append the bound arrays, offset the events), so it arguably belongs implemented — but
either way it needs an explicit decision, and the frozen list currently has no way to
express one.

---

## Adjacent leaks from the same sweep

`a3_method_sweep.py` ran 39 lifecycle-adjacent calls. Four more leak an internal exception
with no classification row, all **native on the fixed path**:

| call | exception | site |
|---|---|---|
| `set_eeg_reference("average")` | `TypeError: list indices must be integers or slices, not tuple` | `mne/_fiff/reference.py:170` |
| `add_channels([...])` | `AttributeError: 'list' object has no attribute 'ndim'` | `mne/channels/channels.py:741` |
| `time_as_index(0.5)` | `RuntimeError` from `times` | `mne/epochs.py:1025` |
| `savgol_filter(10.0)` | `ValueError: … inhomogeneous shape after 2 dimensions` | numpy coercion |

The other 33 behaved: 17 native, 12 declined with the PR's own `NotImplementedError`, 1
fallback (`to_data_frame`).

---

## Pre-existing, checked and out of scope

**Duplicated `selection` — marking one copy bad drops both.** `epochs[[4, 4, 0]]` gives
`selection == [4, 4, 0]`; `_toggle_bad_epoch` uses `.index(epoch_num)` and `_close_impl`
uses `np.isin`, both matching every copy. **Identical on the fixed path on branch and at
base**, both backends.

**`plt.close(fig)` under Agg does not fire the close event**, so the browser's drop never
runs. Identical on the fixed path (mpl 3.10.8). Harness note only: call `fig._close_impl()`.

---

## Checked and clean

- **`epochs[...]`, 16 index forms × `copy=True`/`copy=False`** — int, negative int, `1:4`,
  `::2`, `::-1`, `3:0:-1`, `-2:`, list, list with duplicates, negative list, boolean mask,
  all-False mask, empty list, int ndarray, `event_id` name, chained `[::2][::-1][1:]`. All
  parallel arrays move together and match the permuted source. Parent never mutated.
  **0 violations.**
- **Metadata selection** — `col > 3`, `col > 100` (empty), `name == 'b'`: correct rows and
  bounds, metadata index follows `selection`.
- **450 subselections × 2 backends** over 5 length sets (incl. `(1,100,250,75,180)`,
  `(3,300,3,300,3)`, `(2,3,2,3,2)`), 3 tmin patterns, sfreq 10/100/512.3, 10 index forms —
  bounds vs permuted source, then `plot()` on both backends checking `boundary_times`,
  `epoch_tmins`, `_get_epoch_num_from_time` at every midpoint. **Zero violations.** The
  only 18 exceptions are `epochs[0]` on a 1-sample epoch correctly raising the
  two-time-points guard — the mirror image of P3.
- **Non-contiguous selection → mark → close**, 10 scenarios × 2 backends, including marking
  **all** epochs and marking down to 0. `_get_epoch_num_from_time` reported the `selection`
  number, not the position, at every probe; after `_close_impl` the dropped set was exactly
  the marked set and all parallel arrays kept equal length. **0 violations.**
- **Browsing a reordered object** — `[::-1]`, `[3,1]`, `[4,0,2]`, `[::2]`, boolean mask,
  single: boundary model correct, loaded window bit-equal to that epoch's own samples, axis
  epoch numbers follow the object's order rather than sorted, on both backends.
- **`shift_time`, 126 configurations** — sfreq {10,100,250,512.3,1000,2048} × tmin
  {all-zero, mixed-sign, all-negative} × (relative, tshift) incl. ±0.3, 1e-4, 123.456, −7.5.
  Lengths, `_n_times_per_epoch`, inter-epoch offsets (to 1e-12) all preserved; for
  `relative=False` the earliest epoch lands exactly on `tshift`. **0 problems.** The
  `tshift - _tmin_per_epoch.min()` anchor is the only reading consistent with per-epoch
  `tmin`; worth a docstring sentence, not a defect.
- **`pick`/`drop_channels`/`reorder_channels` aliasing.** `self._data[:] = [...]` is safe:
  `copy()` is `deepcopy`, so a copy taken beforehand keeps its own list and arrays;
  `take()` allocates. Verified by comparing every array of a snapshot after the pick.
- **`copy()`** — new list, new arrays, `_variable_duration` preserved; mutating the copy
  leaves the original untouched.
- **`equalize_event_counts`** — `mintime` and `truncate` drop the right epochs; correctly
  absent from all three raise lists. (Inherits A3-1's stale axis.)
- **Harness invariants I0–I12** over the six `specs.DROPPED` specs × 2 backends × 3 action
  scripts × 3 `PLOT_KWARGS`: **0 failing runs of 72.**
