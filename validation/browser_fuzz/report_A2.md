# A2 — matplotlib display state: decim, scalings, channels, butterfly, events, colours

**Verdict: one new defect in three symptoms, all reachable only because this PR admits a
1-sample epoch. Both seed suspects are killed: S7 (`decim` phase drift) is pre-existing and
bit-identical to base; S8 (`_compute_scalings`) composes correctly on every reachable path.
648 matrix cells × 2 paths produced zero ragged-only violations.**

Harness in `a2/` (`common.py`, `display_checks.py`, `s1`–`s7`, `base_decim.py`, `shots/`).
Base checks used self-contained scripts against `D:/tmp/mne-base`. No repo file modified;
the candidate fix was validated by monkeypatch.

---

## A2-1 — a 1-sample epoch is never drawn as itself

**Verdict: new defect, ragged-only, matplotlib-only.** Qt unaffected. Three symptoms, two
adjacent lines, one fix. Repro: `python -m validation.browser_fuzz.a2.s7_one_sample`.
Independently re-verified by the coordinator.

### Symptom 1 (silent) — marking a 1-sample epoch bad changes nothing on screen

Epochs `(100, 1, 100)` @100 Hz, `n_epochs=2`, epoch 1 marked bad:

| | `epoch_ix` built by `_draw_traces` | painted bands, channel 0 |
|---|---|---|
| ragged `(100,1,100)` | `[0]` — **the view holds epochs 0 and 1** | one band `[0.010, 1.000]`, black |
| fixed `(100,100,100)` | `[0, 1]` | black, then **red** |

Red-pixel counts over the whole canvas (9×3 in, 110 dpi):

| figure | red pixels |
|---|---|
| `a2/shots/a2_ragged_1samp_mid_epoch1_bad.png` | **0** |
| `a2/shots/a2_fixed_control_epoch1_bad.png` | 537 |
| `a2/shots/a2_ragged_1samp_first_epoch0_bad.png` | **0** |
| `a2/shots/a2_fixed_control_epoch0_bad.png` | 513 |

The user clicks, `mne.bad_epochs` gains the number, the epoch is dropped on close — and the
picture never says so. `epoch_colors` behaves the same: `(100,1,100)` with `[red, green,
blue]` paints only red. **Conversely**, marking epoch 0 in `(100,1,100)` paints epoch 1's
only sample red — a *good* epoch drawn as bad.

### Symptom 2 (silent) — a 1-sample epoch first in the view paints nothing

| data | view | painted | result |
|---|---|---|---|
| `(1,100,100)`, `n_epochs=2` | 0:2 | `[0.010, 1.000]` | **epoch 0 has 0 painted samples** |
| `(100,1,100,100)`, one `right` | 1:3 | `[1.010, 2.000]` | **epoch 1 has 0 painted samples** |
| `(100,2,100)` control | 0:2 | both bands | fine — needs exactly 1 sample |

The epoch keeps its boundary lines and its axis number. It just has no data under them.

### Symptom 3 (loud) — `IndexError` when the view is exactly one 1-sample epoch

```
IndexError: index 0 is out of bounds for axis 0 with size 0
  mne/viz/_mpl_figure.py:2268 in _draw_traces
```

`(100,1,100,100)` `n_epochs=1` + one `right` → raises; `(1,100,100,100)` `n_epochs=1` →
raises; `(100,100,100,100)` `n_epochs=1` → fine.

**This is the same line P3 reports, but P3's repro and implied fix do not cover it.** P3 is
about the two-time-points guard moving from per-epoch to total; here the total is 301
samples and only one epoch is short. Restoring a per-epoch guard would reject legitimate
ragged data. The fix has to be at the draw site.

### Fixed-path verdict

All three symptoms need an epoch with exactly 1 sample, which base refuses outright
(`ValueError: Data from at least two time points are required…`, verified on `c4f5ba1e9`).
The branch opens the same data. `force_fixed=True` with `(1,1,1,1)` shows symptoms 2 and 3,
but that configuration is itself only reachable via the guard move P3 reports; with any
legitimate fixed data (`n_times ≥ 2`) the fixed path is clean at every step.

Qt draws all 101 samples of `(100,1,100)` including the first, and opens `n_epochs=1` on a
1-sample epoch without raising. Matplotlib-only, like P1 and P3.

### Cause and fix

`_mpl_figure.py:2266-2267` rebuilds the visible-epoch list by searchsorting the *time
range* instead of asking the view what it holds:

```python
time_range = (self.mne.times + self.mne.first_time)[[0, -1]]
epoch_ix = np.searchsorted(self.mne.boundary_times, time_range)   # 2266
epoch_ix = np.arange(epoch_ix[0], epoch_ix[1])                    # 2267
epoch_nums = self.mne.inst.selection[epoch_ix[0] : epoch_ix[-1] + 1]   # 2268
```

`time_range[1]` is the last *sample's* time, `boundary_times[ix1-1] + (L-1)/sfreq`. When
the last visible epoch has `L == 1` that equals `boundary_times[ix1-1]` exactly,
`searchsorted` (side left) returns `ix1-1`, and the epoch drops out of the colour model.
When it is also the only visible epoch, `epoch_ix` is empty and line 2268 raises.

`_mpl_figure.py:2337` is the second half — `_mask = np.logical_and(_start < this_times,
this_times <= _stop)` — so each colour band starts and ends one sample late. For `L ≥ 2`
cosmetic and pre-existing; for `L == 1` it moves the epoch's only sample into its
neighbour's band. Both lines are byte-identical to base; they are newly *reachable*, like
P3.

Validated by monkeypatch:

```python
epoch_ix = np.arange(*self._get_epoch_ix_range())          # 2266-2267
_mask = np.logical_and(_start <= this_times, this_times <= _stop)   # 2337
```

| case | branch | patched |
|---|---|---|
| `(100,1,100)` epoch 1 bad | 1 band, black | black + **red at x=1.000** |
| `(100,1,100)` colours R/G/B | red only | red + **green at x=1.000** |
| `(1,100,100)` colours | epoch 0 unpainted | both painted |
| `n_epochs=1` on the 1-sample epoch | `IndexError` | opens, paints its sample |
| fixed `(100,)×3` colours | 1 sample unpainted | **0 unpainted** |
| `reference_fixture` colours | 1 sample unpainted | 0 unpainted |

The `<=` makes each boundary sample belong to both neighbouring groups — drawn twice, so
the connecting segment across the boundary is preserved and the window's first sample stops
being dropped. The `epoch_ix` line alone kills symptom 3; symptoms 1 and 2 need 2337 too.

**Decision for @drammock:** if 1-sample epochs are meant to be rejected (P3's other
branch), all three symptoms disappear and only the pre-existing mask shift remains.

---

## S7 — display `decim` phase across boundaries: **killed, pre-existing**

`_mpl_figure.py:2306` decimates the concatenated strip, so epoch *k*'s drawn grid starts at
within-epoch offset `(start − boundary_samples[k]) mod decim`.

**(a) Ragged drifts** whenever a length is not a multiple of `decim`. `(101,103,107,109)`,
first drawn within-epoch index:

| decim | e0 | e1 | e2 | e3 |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | **1** | 0 | **1** |
| 4 | 0 | **3** | 0 | **1** |
| 8 | 0 | **3** | **4** | **1** |

**(b) The fixed path does the same** when `n_times % decim != 0` — `L=101 × 4`, decim 4:
phases `0, 3, 2, 1`. Equal lengths on the *list* path give byte-identical phases to
`force_fixed`, so the ragged code path is not involved.

**Base vs branch, fixed path** (`a2/base_decim.py`, run under both): identical in every
case, e.g. `L=101 decim=4 → {0:0, 1:3, 2:2, 3:1}` on both.

**(c) Pre-existing.** Also reached by realistic `decim="auto"`: 601 samples @1000 Hz with
`lowpass=40` gives `decim=8`, phases `{0:0, 1:7}` on ragged **and** on `force_fixed`.

Two consequences, both reproduced on the fixed path and therefore pre-existing: an epoch
can be drawn on a different grid depending on where you scrolled from; and an epoch can
vanish entirely (`force_fixed (2,2,2,2)` decim 4). **Ragged makes short epochs likelier,
not newly possible.**

**x/y pairing is never broken** — every drawn `y` is an exact affine image of the source
sample at the `x` it is plotted at (max relative residual < 1e-6). Decimation changes
*which* samples are shown, never shows one at the wrong time.

---

## S8 — `scalings`: **killed, no defect**

`_compute_scalings` vs the IQR of the concatenated source strip:

| data | types | relative error |
|---|---|---|
| ragged `(3,300,3,300)` (100:1 spread) | eeg | **0** |
| fixed `(300,)×4` | eeg | 0 |
| ragged `(100,250,75)` 16 ch | eeg, grad, mag, misc, stim | **0** each |
| ragged 5 epochs, `drop=(1,3)` | eeg | **0** |

Drawn amplitude vs `source_ptp / (2 × scalings[type]) × scale_factor`: all `rel < 1e-15`,
for `auto`, `dict(eeg=20e-6)`, and a four-type dict, ragged and fixed.

**The `longest`-sized subset path is dead code for ragged input** — variable-duration
epochs must be preloaded. For *fixed* non-preloaded epochs `_n_times_per_epoch` returns
`len(self.times)`, so the change is a no-op, verified numerically: base and branch both
give `auto eeg = 1.3254721e-06`, matching the full-data IQR.

**Whitening.** Full-rank EEG `noise_cov`: drawn ptp matches
`(projector @ source, DC-removed) / (2 × scalings["whitened"])` to `rel = 0` on ragged, on
the fixed control, and on the 100:1 spread. `w` toggles both ways and returns bit-identical
y data.

---

## Checked and clean

All rows ran ragged **and** an equal-duration control from the same generator, under I0–I12
plus six new display checks (`a2/display_checks.py`).

- **The plot-argument × channel matrix — 648 cells, 1296 figures, zero violations, zero
  errors.** 8 data configs (1/2/3/64/306 channels, mixed types, 3 in `info["bads"]`, 100:1
  spread) × 9 plot-kwarg sets × 9 key scripts.
- **`butterfly` and `b`** including toggling mid-scroll at 64 and 306 channels; `butterfly`
  + `epoch_colors` gives four distinct bands at the four epochs.
- **Clicking channel names** via `fig._click_ch_name`, at 1/2/3/64/306 channels and mixed
  types with pre-set bads: mark, unmark, remark; colour survives `right` and `b`.
- **`pageup`/`pagedown`** at every channel count; **`picks`** by name/type/index;
  **`order`** reversed and permuted; **`group_by`** original/type/selection/position on 306
  Neuromag channels.
- **`events=True` — 12 configurations, all exact**, including multiple `event_id` codes,
  overlapping epochs where one event is drawn in three epochs, and a scrolled `n_epochs=1`
  window. Extends A8's B5, which covered one config.
- **`n_channels` corners**: 1 of 306, 306 of 306, 400 requested of 306, 10 of 3, 5 of 1.

### Two pre-existing items worth one line in the PR description

- **Event bounds at the extremes: the ragged branch is *correct* where the fixed branch is
  not.** An event on an epoch's last sample is included by ragged, excluded by fixed; an
  event past the epoch's end is excluded by ragged, included by fixed (drawing a line at
  x = 1.30 for an epoch spanning `[0, 1.0)`). Both fixed-path errors reproduce
  byte-identically on base. **The PR quietly fixes them on the ragged path only, so the two
  paths now disagree.**
- **The window's first sample is never painted** and every colour band starts and ends one
  sample late (`(100,)×3`: bands `[0.01,1.00]`, `[1.01,1.99]`). **Identical on base.** Only
  listed because it is the second half of A2-1's cause.
