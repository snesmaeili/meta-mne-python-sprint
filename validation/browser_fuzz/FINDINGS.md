# Findings — variable-duration browser sweep

Verdicts follow the triage rule: a behaviour that reproduces with **equal-duration**
epochs is pre-existing and out of scope for PR #14210, however much it deserves a fix
elsewhere. Every entry below was checked both ways.

Run everything with:

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg PYTHONPATH="D:/mne-python;D:/meta-mne-python-sprint" python -m validation.browser_fuzz.fuzz --backend qt
```

---

## A3-1 — subselecting or dropping epochs leaves the time axis at the parent's span

**Verdict: new defect, ragged-only. Silent wrong data, and it corrupts the PR's own
flagship diagnostic.** Independently re-verified. Full detail in `report_A3.md`.

`_getitem` moves `_tmin_per_epoch`/`_tmax_per_epoch` with the epochs but never re-derives
`_raw_times`, which `__init__` built as the *union* of those bounds and which `as_fixed()`
reads back. **`crop()` does re-derive it** — the two subsetting paths disagree.

Four epochs 100/250/75/180 @100 Hz, `tmin=[0,-0.2,0.1,-0.5]`, then `epochs[0]`:

| | value | should be |
|---|---|---|
| holds | 1 epoch, 100 samples, `[0.00, 0.99]` | — |
| `_raw_times` | `[-0.50, +2.29]`, **280 samples** | 100 samples |
| `as_fixed()` | `(1,3,280)`, **540 of 840 NaN** | `(1,3,100)`, 0 NaN |
| `n_contributing == 0` | **180 of 280 time points** | 0 |
| `to_data_frame()` | **280 rows** | 100 |

`n_contributing` exists precisely so a reader can see how many epochs back each time
point. Here it reports 180 points backed by nothing, because they belong to epochs that
were dropped.

**The fallback warning contradicts its own result in one call** — it says "padded to span
0 to 0.99 s" (from the correctly-updated `tmin`/`tmax`) while `as_fixed()` padded to
−0.5 → 2.29 s.

**The browser reaches it**: mark epochs bad, close, `_close_impl` → `drop` → `_getitem`.

**A3-1b:** a subset can never be recognised as fixed-duration again, because `_getitem`
never re-collapses `_variable_duration` the way `crop()` does. Two equal-length epochs
selected out of a ragged set still raise, with a self-refuting message:
`These 2 epochs have durations from 0.990 to 0.990 s, so there is no time axis they share.`

Fix verified by monkeypatch (re-derive `_raw_times` from the sliced bounds, guarding the
empty selection); whether `_getitem` should also re-collapse `_variable_duration` is a
decision, not a silent patch — it just should not differ from `crop`.

---

## A2-1 — matplotlib: a 1-sample epoch is never drawn as itself

**Verdict: new defect, ragged-only, matplotlib-only.** Three symptoms, one cause. Full
detail in `report_A2.md`; independently re-verified.

`_draw_traces` rebuilds the visible-epoch list by searchsorting the *time range*
(`_mpl_figure.py:2266`) instead of asking `_get_epoch_ix_range()`. When the last visible
epoch holds one sample, its last sample's time equals its own boundary, `searchsorted`
returns one short, and the epoch drops out of the colour model. Confirmed directly: for
`(100, 1, 100)` the view holds epochs `(0, 2)` but `_draw_traces` builds `[0]`.

1. **Marking a 1-sample epoch bad changes nothing on screen** — 0 red pixels against 537
   in the equal-duration control. Conversely, marking epoch 0 paints epoch *1* red.
2. **A 1-sample epoch first in the view paints nothing at all**, while keeping its boundary
   lines and axis number.
3. **`IndexError` when the view is exactly one 1-sample epoch** — the same line as P3, but
   **P3's fix does not cover this**: here the total is 301 samples and only one epoch is
   short, so restoring a per-epoch guard would reject legitimate ragged data. The fix must
   be at the draw site.

Fix verified by monkeypatch: `epoch_ix = np.arange(*self._get_epoch_ix_range())`, plus
`_start <= this_times` at line 2337 for symptoms 1–2. Qt is unaffected.

**Decision for @drammock:** if 1-sample epochs are meant to be rejected (P3's other
branch), all three symptoms disappear.

---

## S7 and S8 — both killed

- **S7 (`decim` phase drift across boundaries): pre-existing.** Ragged drifts whenever a
  length is not a multiple of `decim` — but so does the fixed path when
  `n_times % decim != 0`, and base and branch give **byte-identical phases** on the fixed
  path in every case tested. Ragged makes short epochs likelier, not newly possible. x/y
  pairing is never broken: every drawn `y` is an exact affine image of the source sample at
  the `x` it sits on.
- **S8 (`_compute_scalings`): no defect.** Relative error **0** against the concatenated
  source IQR on ragged, fixed, mixed channel types, and a 100:1 duration spread. The
  `longest`-sized subset path is dead code for ragged input (must be preloaded) and a
  verified no-op for fixed.

---

## Pre-existing, worth one line in the PR description

**Event bounds at the extremes: the ragged branch is correct where the fixed branch is
not.** An event on an epoch's last sample is included by ragged and excluded by fixed; an
event past the epoch's end is excluded by ragged and included by fixed. Both fixed-path
errors reproduce byte-identically on base — so **the PR quietly fixes them on the ragged
path only, and the two paths now disagree.**

---

## F4 — Qt: a one-sample epoch's vline is clamped just outside its own epoch

**Verdict: new defect, ragged-only. Found while verifying the F1 fix, not by any slice.**

`VLine` bounds are set as `bmax = boundary_times[k + 1] - 1 / sfreq`. For an epoch holding
a single sample that should equal `bmin` exactly — but in float64 it lands a hair below:
lengths `(137, 1)` at 100 Hz give `bmin = 1.37` and `1.38 - 0.01 = 1.3699999999999999`.
The line is then clamped to a position *before* its own epoch's start, and every latency
readout attributes it to the previous epoch.

```
_add_vline(0.0) on lengths (137, 1)
  -> I8 vlines sit at different latencies: [0.0, 1.37]
```

Both lines are placed at latency 0.0, which is correct; only the clamp moves one of them.

Fixed at all four bound-setting sites in `_pg_figure.py` with
`bmax = max(bmax - 1 / sfreq, bmin)`.

This is why the first fix pass left 30 residual ragged failures in the sweep. After it,
a clean 1320-run sweep reports **zero**.

---

## A3-2 / A3-3 — `drop_bad(reject=…)` and `concatenate_epochs` leak internal errors

**Verdict: new defects, ragged-only.** Neither is on any of the three classification
lists, so the user gets an internal `RuntimeError` about a time axis instead of one of the
PR's three deliberate messages.

- **`drop_bad(reject=…)` / `flat=…`** reaches `Epochs.times` via `_handle_tmin_tmax`. The
  ragged implementation **already exists** (`_load_variable_from_raw` rejects per epoch
  using `_n_times_per_epoch`) but is unreachable, since its only caller is the
  non-preloaded path that ragged epochs forbid. The same rejection at *construction* works
  correctly. Note a blanket raise-list entry would also break the no-arg `drop_bad()`,
  which `_concatenate_epochs` calls internally.
- **`concatenate_epochs`** fails at `np.allclose(epochs.times, …)` and its message names
  only the second list element. Structurally unclassifiable: it is a module-level
  function, so the three `setattr` loops cannot reach it however the dicts are edited.

Four more leak the same way, all native on the fixed path: `set_eeg_reference("average")`,
`add_channels`, `time_as_index`, `savgol_filter`.

Relevant to @drammock's "freeze the list": the list is frozen, but these are holes *in* it
rather than requests to widen it.

---

## P1 — matplotlib: a click in an epoch's last half-sample deletes the marker and mislabels it

**Verdict: new defect ON THE EQUAL-DURATION PATH. A regression, not a ragged-only issue.**
Found by the parity slice, independently re-verified. Full detail in `report_A7.md`.

The matplotlib `_recompute_epochs_vlines` rewrite has **no `is_variable_duration` guard**,
so ordinary fixed-duration epochs go through the new code. Qt keeps its old
`t % epoch_dur` path behind such a guard and is unaffected.

Four equal-duration epochs, 100 samples at 100 Hz, `n_epochs=2`:

| click x | base (`c4f5ba1e9`) | branch |
|---|---|---|
| 0.990 | 2 lines, `0.99 s` | 2 lines, `0.99 s` |
| 0.995 | 2 lines, `0.995 s` | **0 lines**, `1 s` |
| 0.999 | 2 lines, `0.999 s` | **0 lines**, `1 s` |

The epoch's last sample is at 0.99 s, so `1 s` is a latency the data does not contain, and
it is displayed with no line under it. Dead zone is `0.5/L` of each epoch — 0.5 % at
L=100, **25 % at L=2**.

Fix (verified by monkeypatch), clamping to the clicked epoch's own length so ragged
behaviour is preserved:

```python
n_samp = int(round((boundary_times[clicked_ix + 1] - boundary_times[clicked_ix]) * sfreq))
offset = int(np.clip(round((xdata - boundary_times[clicked_ix]) * sfreq), 0, n_samp - 1))
```

---

## P3 — matplotlib: 1-sample epochs are now admitted and then crash at `n_epochs=1`

**Verdict: new on the equal-duration path.** Resolves seed suspect S10.

The "at least two time points" guard moved from per-epoch (`len(inst.times) < 2`) to the
**total** concatenated count (`self.mne.n_times < 2`), so two 1-sample epochs now pass it.
Base raised a clear `ValueError` on both backends; the branch raises
`IndexError: index 0 is out of bounds for axis 0 with size 0` in matplotlib at
`_mpl_figure.py:2268` — a line byte-identical to base, only newly reachable. Qt opens fine.

---

## P2 — matplotlib: vline latency is now quantised to the nearest sample

**Verdict: fixed-path behaviour change, probably intended — but it should be stated in the
PR rather than discovered.** Base reported the raw click position; the branch snaps to the
sample grid (`0.231 s` → `0.23 s`, `0.018 s` → `0.02 s`). The line moves by at most half a
sample. Defensible, since there is no datum at 0.231 s.

---

## F1 — Qt: scrolling drags a vline to the wrong latency in short epochs

**Verdict: new defect, ragged-only. Confirmed.**

`MNEQtBrowser._xrange_changed` repositions every vline as `bmin + rel_vl_t`, where
`rel_vl_t` is a seconds-offset read off the first line relative to the previously
first-visible epoch's boundary
([_pg_figure.py:1378](D:/mne-qt-browser/src/mne_qt_browser/_pg_figure.py:1378)). That
is the fixed-grid assumption the latency model replaced: it never converts through
`latency_at`, and it never calls `_set_epoch_vline_visibility`. When the target epoch
is shorter than the offset, `setValue` clamps the line to the epoch's own `bmax`, so
the line stays **visible at a latency it was never placed at** instead of being hidden.

Repro — epochs of 100/250/75/180 samples at 100 Hz, boundaries `[0, 1, 3.5, 4.25, 6.05]`:

| | ragged | fixed (100×4) |
|---|---|---|
| after `_add_vline(0.9)` | offsets `[0.9, 0.9]` | offsets `[0.9, 0.9]` |
| after `hscroll("right")` | offsets **`[0.9, 0.74]`** | offsets `[0.9, 0.9]` |

0.74 s is `4.24 - 3.5`, i.e. the 75-sample epoch's last sample: the line was clamped,
not hidden. The fixed path is unaffected, so this is the PR's to fix.

Script: `scratchpad/qt_vline_repro.py`.

---

## F3 — both backends: the epoch histogram ('h') crashes on ragged epochs

**Verdict: new defect, ragged-only. Confirmed on matplotlib and Qt.**

`BrowserBase._create_epoch_histogram`
([mne/viz/_figure.py:713](D:/mne-python/mne/viz/_figure.py:713)) computes

```python
ptp = np.ptp(epochs.get_data(copy=False), axis=2)
```

`get_data()` returns a **list of per-epoch arrays** when durations vary, so `np.ptp`
raises `ValueError: setting an array element with a sequence. The requested array has an
inhomogeneous shape after 2 dimensions.` Qt re-raises it as a `RuntimeError` from inside
the event loop, which hides the cause entirely.

| | ragged (`reference_fixture`) | fixed (`equal_fixed_path`) |
|---|---|---|
| matplotlib, `h` | `ValueError` | no exception |
| qt, `h` | `RuntimeError` wrapping it | no exception |

This is in MNE-Python's shared `BrowserBase`, not in either backend, so it is the PR's to
fix. The PR classifies 16 public methods for ragged input but the browser's own histogram
path was not among them.

Peak-to-peak is a per-trial reduction and is perfectly well defined for ragged epochs, so
the fix is to compute it per epoch rather than to decline:

```python
data = epochs.get_data(copy=False)
ptp = np.array([np.ptp(d, axis=-1) for d in data])   # works for list and ndarray
```

Repro:

```python
from validation.browser_fuzz import specs
from validation.browser_fuzz.build import build
from validation.browser_fuzz.runner import open_browser
case = build(specs.BY_NAME["reference_fixture"])
fig = open_browser(case, "matplotlib", n_epochs=2)
fig._fake_keypress("h")   # ValueError
```

---

## F2 — Qt: the vline list is not resized when the epoch count changes

**Verdict: pre-existing (reproduces on the fixed path), but newly load-relevant.**

After `change_duration(step=1)`, `mne.epoch_idx` grows to 3 entries while `mne.vline`
still holds 2. Identical on ragged and fixed epochs, so it predates this PR.

It matters more now: the ragged visibility logic (`_set_epoch_vline_visibility`,
`_epoch_vline_state`) `zip`s those two lists, so a stale length silently drops the
visibility decision for the epochs past the end of the list. Worth raising with
@drammock as a decision — fix here, or file separately.

---

## V3 — Qt: epoch numbers vanish from the axis while `get_labels()` still reports them

**Verdict: pre-existing mechanism, newly reachable in ordinary configurations on the
ragged path.** A8 first scored this ragged-only; the 50- and 100-epoch fixed controls
corrected it.

pyqtgraph drops any tick label whose text rect is not fully inside the axis
`boundingRect` (`AxisItem.generateDrawSpecs`). `TimeAxis.get_labels()` — the accessor
`tests/test_pg_specific.py` asserts on — keeps returning every label. So a test on the
accessor passes while the screen is short a label. That is exactly the failure mode the
pending mne-qt-browser AGENTS.md warns about.

Measured rects for `(3, 300, 3, 300)`, `n_epochs=4`: at 1600 px label `'0'` occupies
`[0.2, 7.2]` and is painted; at 900 px it occupies `[-1.5, 5.5]` and is not.

| data | n_epochs | 1600 px | 900 px | 420 px |
|---|---|---|---|---|
| `(3,300,300,300)` | 4 | drops `'0'` | drops `'0'` | drops `'0'` |
| `(300,300,300,3)` | 4 | drops `'3'` | drops `'3'` | drops `'3'` |
| `(3,300,3,300)` | 4 | ok | drops `'0'` | drops `'0'` |
| `(3,3,300,300)` | 4 | `'0'`/`'1'` 0.5 px apart, reading as `01` | drops `'0'` | drops `'0','1'` |
| fixed `(150,)×4` | 4 | ok | ok | ok |
| fixed `(60,)×50` | 50 | ok | ok | drops `'0','49'` |
| fixed `(60,)×100` | 100 | ok | drops `'99'` | drops `'0','98','99'` |

The fixed path reaches it, so the mechanism predates the PR. What ragged changes is the
cost of admission: **4 epochs at ordinary window widths**, versus 50–100 crammed into one
window. Same category as F2 — pre-existing but newly load-relevant, a decision for
@drammock rather than a blocker.

Worse than incomplete in one case: at 900 px with `(3,3,300,300)` the leftmost number
painted is `'1'`, sitting over pixels that belong to epoch 0.

A8 validated a candidate fix by monkeypatch: clamp the label inside the axis rect rather
than dropping it, since an epoch number identifies a region, not a coordinate. Patch
`TimeAxis.generateDrawSpecs` **only** — patching `AxisItem` also catches `ChannelAxis`,
whose `tickStrings` then indexes past the channel list.

---

## N1 — matplotlib: `_xtick_formatter` crashes when the axis has fewer than two ticks

**Verdict: pre-existing. Out of scope.**

`np.diff(self.mne.ax_main.get_xticks())[0]`
([_mpl_figure.py:2085](D:/mne-python/mne/viz/_mpl_figure.py:2085)) raises `IndexError`
when matplotlib puts fewer than two ticks on the axis, which happens for short views
under Agg. Reached by any click that shows a vline.

Reproduces with equal-duration epochs (`lengths=(100, 100)` and `(1000, 1000)`,
`force_fixed=True`); does **not** depend on ragged input. Note it does not reproduce
with four equal epochs — it is about the tick count, not the durations.

---

## N2 — RESOLVED, and I had it wrong

**Superseded by A4-1. The original verdict below was mistaken.**

I wrote that a non-boundary `setXRange` "reproduces identically on the fixed path" and so
was pre-existing. That is true only of a *synthetic* `setXRange` called with arbitrary
numbers. It is not true of the ranges the widgets actually produce: those are aligned on
the fixed path and unaligned on the ragged one, because `OverviewBar._set_range_from_pos`
and `TimeScrollBar._time_changed` pick a start epoch and then reuse the current
`mne.duration` as the width — and with ragged epochs the duration of *n* epochs depends on
*which* *n*.

A real gesture reaches it: an overview-bar click via `QTest.mousePress` leaves up to 27%
of the window blank while the epoch tick labels still name the epochs it claims to show.
See A4-1 in `report_A4.md`.

The viewbox itself cannot be pushed off a boundary — drags outside annotation mode are
ignored, the wheel routes to `hscroll`, and there is no zoom path. It is the widgets
*around* the plot that set the range without consulting the boundary model.

Consequence for the harness: `ov_click:*` and `hscroll_bar:*` belong in the default
alphabet. `setxrange:*` can stay opt-in.

