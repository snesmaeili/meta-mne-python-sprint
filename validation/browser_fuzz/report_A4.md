# A4 — Qt navigation and vertical lines

Harness addition: `qt_nav.py` — actions `vline_at:<latency>`, `resize:WxH`,
`crosshair:<frac>`; invariants **N1** (view cache vs `viewbox.viewRange()`), **N2** (a
visible vline's label reads its own latency), **N3** (one vline per visible epoch). Kept
in its own module so it does not collide with `actions.py` / `invariants.py` /
`qt_widgets.py`.

All commands assume
`QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg PYTHONPATH="D:/mne-python;D:/meta-mne-python-sprint"`.
Every fix below was validated by monkeypatch; no edits were made to either mne repo.

---

## A4-1 — an overview-bar click, or a scrollbar move, shows a window that is not the epochs it claims

**Verdict: new defect, ragged-only.** Clean against `force_fixed=True` (100×4) and against
equal-length epochs built on the list path. **This also answers N2** — see the correction
in `FINDINGS.md`.

`OverviewBar._set_range_from_pos` ([_widgets.py:987](D:/mne-qt-browser/src/mne_qt_browser/_widgets.py:987))
picks a target epoch, converts it to `boundary_times[epo_idx]`, then keeps the **current**
`mne.duration` as the width:

```python
x = self.mne.boundary_times[epo_idx]
xmin = np.clip(x, 0, self.mne.xmax - self.mne.duration)      # line 1007
xmax = np.clip(xmin + self.mne.duration, self.mne.duration, self.mne.xmax)
```

`TimeScrollBar._time_changed` ([_widgets.py:1276](D:/mne-qt-browser/src/mne_qt_browser/_widgets.py:1276))
does the same. With ragged epochs the duration of *n* epochs depends on **which** *n*, and
`xmax - duration` is not a boundary either, so both ends can land mid-epoch.

Epochs 100/250/75/180 at 100 Hz, boundaries `[0, 1, 3.5, 4.25, 6.05]`:

| gesture | n_epochs | view range | loaded | result |
|---|---|---|---|---|
| click far right | 2 | `[2.550, 6.050]` | `[3.50, 6.04]` | **0.95 s (27.4%) of the window blank** |
| scrollbar → 1 | 2 | `[1.000, 4.500]` | `[1.00, 4.24]` | 0.26 s (7.4%) blank |
| click far right | 3 | `[1.800, 6.050]` | epochs 2:4 | t_start mid-epoch-1 |
| click 25% | 1 | `[1.000, 2.000]` | `[1.00, 3.49]` | **1.49 s of a 2.5 s epoch off-screen** |
| end, then click far left | 2 | `[0.000, 2.550]` | `[0.00, 3.49]` | **26.9% of loaded samples drawn outside the view** |

Fixed control gives `[0,2]`, `[1,3]`, `[2,4]` — always aligned, never blank. Epoch tick
labels still read `2 3` while a quarter of the plot is empty, and
`_get_epoch_ix_range()` / `_get_start_stop()` still report whole epochs, so nothing
announces the mismatch.

**Additional consequence found visually by A8 (V2):** at `n_epochs=1` the same gesture
(click 25% along the bar) gives view `[1.000, 2.000]` inside epoch 1, which spans
`[1.000, 3.490]`. `TimeAxis.tickValues` selects midpoints inside the view, epoch 1's
midpoint (2.25) falls outside it, so `get_labels()` returns `[]` and **the axis carries no
epoch number at all** — a window showing 40% of one epoch, unlabelled, with no separator
in sight. Fixed control at the same click: view `[0.000, 1.500]`, label `['0']`.
See `shots/a8_ovclick_ragged_3_one_epoch_25pct.png`.

**Fix.** Both sites should ask `epoch_window` for the window instead of reusing
`mne.duration`; it already clamps `start_ix`, so the two `np.clip` lines go with it, and
the fixed case degenerates to the old numbers — one branch covers both.

- `_widgets.py:992–1008`, inside `if self.mne.is_epochs:` — replace `x = ...` and the two
  clips with `xmin, dur = epoch_window(self.mne.boundary_times, epo_idx, self.mne.n_epochs)`;
  `xmax = xmin + dur`. Leave the raw branch alone.
- `_widgets.py:1278–1283` — for epochs,
  `t_start, duration = epoch_window(self.mne.boundary_times, int(value), self.mne.n_epochs)`
  then `setXRange(t_start, t_start + duration, padding=0)`.

Verified: every overview click (0/50/100% of the bar) and every scrollbar value then lands
on a boundary at both ends, across `(100,250,75,180)` at n_epochs 1/2/3, `(100,50,250,180)`,
equal lengths with per-epoch `tmin`, and the fixed control (numbers unchanged).

---

## A4-2 — a vline's label reads a latency the line is not at

**Verdict: new defect, ragged-only.**

`VLineLabel.valueChanged` ([_graphic_items.py:962](D:/mne-qt-browser/src/mne_qt_browser/_graphic_items.py:962))
begins `if not self.isVisible(): return`. `_add_vline` moves every line first and only then
calls `_set_epoch_vline_visibility`, so a hidden line gets its new position while the label
is dropping updates, then is made visible carrying the old text. `setVisible` emits no
value change, so it never refreshes. Fixed-duration epochs never hide a line.

Repro is the repo's own `test_variable_duration_vline_positions` sequence:

| slot | epoch | x | true latency | label shown |
|---|---|---|---|---|
| 0 | 0 | 0.500 | **0.500 s** | `0.000 s` |
| 1 | 1 | 1.500 | 0.500 s | `0.500 s` |
| 2 | 2 | 4.000 | **0.500 s** | `0.000 s` |
| 3 | 3 | 4.750 | 0.500 s | `0.500 s` |

Two of four labels read `0.000 s` for lines at 0.500 s. Fixed control: all four read
`0.500 s`.

**Fix.** `_pg_figure.py:1257–1258` — refresh the label of any line going hidden → visible:

```python
for vl, visible in zip(self.mne.vline, mask):
    was = vl.isVisible()
    vl.setVisible(bool(visible))
    if visible and not was:
        vl.label.valueChanged()   # the label skips updates while it is hidden
```

---

## A4-3 — F1 extended: three things it does not yet say

**(a) The corruption is permanent.** `rel_vl_t` is re-read from `vline[0].value()` on
*every* range change (`_pg_figure.py:1385–1388`), so a clamped value becomes the next
scroll's reference offset:

| step | latencies |
|---|---|
| after `_add_vline(0.9)` | `[0.900, 0.900]` |
| `hscroll("right")` | `[0.900, 0.740]` ← clamped in the 75-sample epoch |
| `hscroll("right")` | `[0.740, 0.900]` |
| `hscroll("left")` ×2, back at the first view | **`[0.740, 0.740]`** — 0.900 is gone |

**(b) Visibility flags ride on the list slot, not the epoch.** `_xrange_changed` never calls
`_set_epoch_vline_visibility`. With `lengths=(100,50,250,180)`, n_epochs=2, vline at 0.9 s,
one scroll: slot 0 sits on a 50-sample epoch that never reaches 0.9 s and is **visible** at
0.490; slot 1 sits on a 250-sample epoch that does reach it and is **hidden**.

**(c) No clamping is needed — the tightest repro is one line, one scroll, equal lengths.**
`bmin + rel_vl_t` is an offset from the *boundary*; the latency is measured from the
*event*. With four equal-length epochs and per-event `tmin` `[0, -0.2, 0.1, -0.5]`,
`variable_duration` is `True`, nothing clamps, and the latency walks
0.500 → 0.300 → 0.600 → 0.000. Fixed control holds +0.500.

**The overview bar is not implicated**: `OverviewBar.update_vline`
([_widgets.py:934](D:/mne-qt-browser/src/mne_qt_browser/_widgets.py:934)) is
`if self.mne.is_epochs: pass`, so there is no epoch vline there at all — pre-existing and
identical on the fixed path.

**Fix — two changes, not one.**

1. **Carry the latency in state, not in a position.** Every line can be hidden at once, and
   then no position on screen carries the latency; deriving it from `vline[0]` there
   silently resets it to the epoch start. In `_add_vline` and `_vline_slot`, on the variable
   path set `self.mne.vline_latency = latency_at(...)`.
2. **Reposition by latency and re-apply visibility.** After
   `self.mne.epoch_idx = np.arange(*boundary_idxs)` (line 1393), call `latency_positions(...)`,
   park non-reaching epochs at their own `boundary_times[ix]`, and per line do
   `setBounds`, `setValue`, `setVisible`, `vl.label.valueChanged()` (A4-2). Keep the
   `rel_vl_t` loop for the fixed path. This is the arithmetic `_epoch_vline_state` already
   does, so factor it as a latency-taking sibling rather than duplicating it.

**Interaction with F2:** the fix still `zip`s `epoch_idx` with `mne.vline`, so F2 must be
fixed too, or a `change_duration` after `_add_vline` keeps dropping the last epoch's line.

---

## A4-4 — crosshair readout overshoots by one sample at the last pixel

**Verdict: new defect, ragged-only, cosmetic.**

`_mouse_moved` clamps the sample index into `mne.times` so the reported *value* is the last
sample's, then converts the raw mouse `x` — not that sample — through `latency_at`
(`_pg_figure.py:1353`), which clamps the epoch index but not the offset within it. Hovering
at `boundary_times[-1]` reports `x=1.800 s` where the last real latency is 1.790.

**Fix.** `_pg_figure.py:1353–1358` — pass `self.mne.times[idx]` instead of `x`, so the x and
y halves of the status bar are consistent by construction.

*Pre-existing, out of scope:* on the **fixed** path this readout is worse —
`rel_idx = idx % len(inst.times)` reports `0.510 s` for a hover at 0.500 s and wraps to
`0.000 s` at 0.990 s. The new `latency_at` path is exact everywhere except the pixel above.

---

## A4-5 — `change_duration` at the end of the data emits a non-boundary range on the way

**Verdict: new defect, ragged-only, transient (self-corrects within the same call).**

`change_duration` calls `ax_hscroll.update_duration()` before its own `setXRange`.
`update_duration` ([_widgets.py:1296](D:/mne-qt-browser/src/mne_qt_browser/_widgets.py:1296))
calls `setMaximum(...)`, which clamps the scrollbar value and emits `valueChanged` while
`external_change` is `False`, so `_time_changed` fires a `setXRange` with the **old**
duration. `_xrange_changed` therefore runs twice, the first time with `(1.000, 3.550)` —
unaligned. Fixed control also fires twice but both ranges are aligned, so the doubled
redraw is pre-existing and the unaligned intermediate is ragged-only.

**Fix.** `update_duration` changes the bar's geometry, not the view, so its
`setMaximum`/`setPageStep` should not drive `_time_changed`. Guard them the way
`update_value` already guards itself, or wrap in `QSignalBlocker(self)`.

---

## Assigned questions

**N2 — can a real gesture leave the range off boundaries? Yes, ragged-only.** Two gestures
(A4-1) plus one internal path (A4-5). The viewbox cannot be pushed off a boundary:
`RawViewBox.mouseDragEvent` ignores drags outside annotation mode, `wheelEvent` routes to
`hscroll`, and `hideButtons()` + `setLimits` leave no zoom path. It is the widgets *around*
the plot that set the range without consulting the boundary model.

**S2 — killed.** `epoch_dur` has exactly two readers, both in the `else` of
`is_variable_duration(mne)`, which reads `mne.inst.variable_duration`; for the epochs
browser `mne.inst` is the plotted `Epochs`, so the flag cannot be `False` while
`boundary_times` is ragged. Confirmed at runtime: `epoch_dur == 1.0` (wrong for three of
four epochs) while the latency path returns the right answers. The one place
`boundary_times` is built without `epoch_tmins` — the ICA-sources path,
[mne/viz/ica.py:1387](D:/mne-python/mne/viz/ica.py:1387) — is S2's exact shape but
unreachable, because `ica.fit()` on ragged epochs fails first. `epoch_dur` is dead weight
worth deleting, not a live defect.

*Adjacent, outside this slice:* `ICA.fit(ragged_epochs)` raises
`AttributeError: 'list' object has no attribute 'shape'` at
[mne/preprocessing/ica.py:842](D:/mne-python/mne/preprocessing/ica.py:842) rather than a
clear `NotImplementedError`.

**S9 — clean.** `_get_x_from_norm` does not exist on this branch; the path is `_mapToData`.
Last pixel correct (`1380 → 3`, `1381 → 3`, `1382 → "+offbounds"`, `-1 → "-offbounds"`),
identical on the fixed path. The negative `epo_idx = len(inst) - n_epochs` is unreachable
because MNE clamps `n_epochs` to `len(epochs)` first. Every epoch line, bad-epoch rect and
viewrange rect sits within half a pixel of `width * t / xmax`.

---

## Coverage that found nothing

- **Navigation sweep**, 357 runs: `hscroll` all four ways and `change_duration(±1)` over all
  11 duration-spread specs and all 6 count specs (1–2000 epochs), at n_epochs 1/2/3, under
  seven action scripts — **no violations**. Every failure in this slice needs a vline or a
  widget.
- **View-range cache**: `mne.t_start`/`mne.duration` **never drifted** from
  `viewbox.viewRange()` across 16 actions and six configurations. The defect is that the
  *authoritative* range goes off-boundary (A4-1), not that the cache disagrees.
- **Resize**: 900×450, 400×300, 160×120, 1600×900, 240×700, with a bad epoch marked and a
  vline placed, on `(100,250,75,180)` and a 50-epoch object at n_epochs 8 and 50 (per-epoch
  widths ≈2 px): all geometry within half a pixel, nothing waiting on a range signal.
  **No ragged-specific staleness** — nothing new in this branch is pixel-positioned, and
  `OverviewBar.resizeEvent` already recomputes from `boundary_times`. (`_resize_by_factor`
  is a `pass` in this backend, so `fig.resize` is the real gesture.) One pre-existing
  butterfly inconsistency noticed, not pursued: `OverviewBar.resizeEvent` lacks the
  butterfly branch `update_viewrange` has.
- **Teardown**: **nothing to add to the `delattr` list.** The branch adds only numpy arrays
  and a float to `self.mne`; no new Qt object. Six open/close cycles × four scripts × two
  paths under `warnings.simplefilter("error")`: zero surviving `MNEQtBrowser` instances
  after `gc.collect()`, no teardown errors, no warnings. The F2 mismatch does not break
  teardown in either direction.
- **`precompute=True`**: 11 spread specs plus 1/3/50-epoch objects — `global_data.shape[-1]`
  equals the true total, no NaN, `global_times` matches the concatenated axis exactly, and
  the crosshair's precomputed branch agrees with the non-precomputed one.
- **Vlines at awkward latencies**: exactly on a boundary, at the first epoch's last sample,
  half a sample past it, over 1/2/3/50/500/2000-epoch objects at n_epochs 1/2/5/10, each
  followed by six scrolls: only the A4-3 patterns. Boundary placement on the ragged path is
  correct. *Pre-existing:* on the **fixed** path `_add_vline(0.0)` clamps the second line
  into epoch 0 at 0.99 s; `git diff 2a9fc9c..e371eb3` confirms that branch is untouched by
  this PR.

## F2 deepened (not a new finding)

**Pre-existing, byte-for-byte identical on the fixed path.** The vline list length is frozen
at creation, so `change_duration(-1)` afterwards leaves lines with no epoch; `_xrange_changed`'s
`zip` never touches them and they stay **visible**. n_epochs=2 → `change_duration(+1)` ×2 →
`_add_vline` → `change_duration(-1)` ×2 → `hscroll("right")` ×2 draws 2, then **3**, then
**4** lines inside a 2-epoch view, with duplicates at the same x.
