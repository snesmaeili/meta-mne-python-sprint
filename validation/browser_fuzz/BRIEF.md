# Brief for corner-case agents — variable-duration epoch browsing

You are hunting corner cases in MNE-Python PR #14210 (branch
`epochs-variable-operations` in `D:\mne-python`) and its companion
mne-qt-browser #452 (branch `variable-duration-epochs` in `D:\mne-qt-browser`).

The PR lets `Epochs` hold trials of different lengths and browse them natively. The
browser's x axis is no longer a uniform grid; it is built from cumulative per-epoch
sample counts:

```
lengths          = per-epoch sample counts
boundary_samples = np.r_[0, np.cumsum(lengths)]
boundary_times   = boundary_samples / sfreq
n_times          = boundary_samples[-1]
```

Every navigation, click, vline, scrollbar and overview-bar path reads that model. Your
job is to find where it is read wrongly.

## Environment — required for every command

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg PYTHONPATH="D:/mne-python;D:/meta-mne-python-sprint"
```

`mne` is **not** editable-installed: without `PYTHONPATH` you silently import the
released 1.11.0 from site-packages and everything skips or passes vacuously. Always
assert `hasattr(mne.EpochsArray, "variable_duration")` before trusting a run.

`mne_qt_browser` *is* editable at `D:\mne-qt-browser\src`, so it needs no path entry.

**If you grab Qt screenshots, `QT_QPA_FONTDIR` is also mandatory:**

```bash
QT_QPA_PLATFORM=offscreen QT_QPA_FONTDIR="C:/Windows/Fonts"
```

Under bare `offscreen` on this machine `QFontDatabase.families()` returns **0 families**
and every string in a grab — channel names, epoch numbers, vline labels, axis titles —
renders as an empty box. The result looks like a working browser with no text anywhere.
Any "the screenshot looked fine" without this is worthless.

Qt runs headless under `QT_QPA_PLATFORM=offscreen` — verified working. Do not open real
windows; several agents run at once.

## The harness

`D:\meta-mne-python-sprint\validation\browser_fuzz\`

| module | what it gives you |
|---|---|
| `build.py` | `Spec` / `build(spec)` → a `Case` carrying `lengths`, `boundary_samples`, `boundary_times`, `source`, `tmins`, all computed from the source arrays and never from the object under test |
| `specs.py` | the data matrix (`ALL`, `FAST`, `BY_NAME`) and `PLOT_KWARGS` |
| `invariants.py` | `check_all(fig, case, backend)` → list of violation strings (I0–I10) |
| `actions.py` | `build_alphabet(fig, case, backend)` → `[(name, callable)]` |
| `runner.py` | `run(spec, backend, action_names)` → `Result` with `.ok` and `.report()` |
| `fuzz.py` | `fuzz_one(spec, backend, seed, n_steps)` → result plus a shrunk minimal sequence |

Example:

```python
from validation.browser_fuzz.runner import run
from validation.browser_fuzz import specs
r = run(specs.BY_NAME["reference_fixture"], "qt", ["vline:0.9", "hscroll:right"])
print(r.report())
```

Extend the harness where your slice needs it — new `Spec` fields, new actions, new
invariants. Keep expectations independent of the code under test: if an invariant asks
the browser what it thinks the boundaries are and compares that to itself, it proves
nothing.

## The triage rule — this is the important part

**A behaviour that also happens with equal-duration epochs is pre-existing and out of
scope.** Before reporting anything, re-run it with:

- `Spec(lengths=(100, 100, 100, 100), force_fixed=True)` — the fixed 3-D path
- and, if you still believe it is new, against the base commit `c4f5ba1e9`
  (worktree at `D:\tmp\mne-base`; put that first on `PYTHONPATH` instead of
  `D:/mne-python`)

Report each finding as:

1. a minimal runnable repro (a `run(...)` call or a short script)
2. the invariant it violates, or the exception
3. observed vs expected, with numbers
4. **verdict**: new defect / pre-existing / harness bug

A finding without a repro does not count. A finding you did not check against the fixed
path does not count.

## Already triaged — do not re-report

Read `FINDINGS.md` first. F1 (Qt vline misplaced by `_xrange_changed` after scrolling),
F2 (vline list not resized on `change_duration`), N1 (`_xtick_formatter` IndexError with
<2 ticks), N2 (`setXRange` to a non-boundary range) are known. Deepen them if your slice
touches them; do not rediscover them.

## Seed suspects (unconfirmed — confirm or kill, and do not stop here)

- **S2** `mne.epoch_dur` (`_pg_figure.py:346`) is set from the *first* epoch only. It is
  meant to be unreachable when durations vary. Find any path that still reads it.
- **S4** `sampling_period = (1/sfreq)/sfreq` (`_figure.py:160`) is a deliberate
  non-quantity used to nudge a `searchsorted`. Is it big enough to beat float error at
  `boundary_times[-1]` with 2000 epochs, and small enough not to skip an epoch at
  sfreq 10?
- **S5** non-integer sfreq (512.3): `boundary_samples` on the derived/ICA path is
  `round(boundary_times * sfreq)` (`_figure.py:205`), which can disagree with the true
  cumulative counts.
- **S6** `_get_epoch_num_from_time` (`_figure.py:553`) searchsorts without clamping; at
  `time == boundary_times[-1]` it indexes past the end.
- **S7** display `decim` is applied to the *concatenated* strip
  (`_mpl_figure.py:2306`), so the decimation phase drifts across boundaries when an
  epoch length is not a multiple of `decim`. Likely pre-existing — the job is to show
  ragged input does not make it worse.
- **S8** `_compute_scalings` (`viz/utils.py:1404`) sizes a random epoch subset by the
  *longest* epoch, then concatenates `inst._data` for the list case. Check the two
  paths compose.
- **S9** `OverviewBar._mapFromData` / `_get_x_from_norm` now go through
  `boundary_times`. Check the last pixel and the empty-`add_chs` case.
- **S10** a 1-sample epoch: the `n_times >= 2` guard is on the *total*, so one 1-sample
  epoch passes it and then has `tmin == tmax`.

## Deliverable

A markdown report: findings ranked with silent-wrong-picture before loud-exception,
each with repro, verdict, and — where you are confident — a suggested fix with the file
and line. Do not edit `D:\mne-python` or `D:\mne-qt-browser`; the harness directory is
yours to extend.
