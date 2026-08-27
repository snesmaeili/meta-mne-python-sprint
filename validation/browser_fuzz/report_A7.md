# A7 — parity guard: equal-duration epochs, pre-PR vs branch

**Verdict: parity holds on the numbers that matter — every sample bound, every loaded
sample, every plotted trace, every boundary array is bit-identical across 7,697 recorded
states. But two real fixed-path behaviour changes were found, both in the matplotlib
vline, one of them a silent wrong picture. Raw and ICA browsing are untouched.**

## Method

`parity.py` is self-contained — it imports nothing from the rest of the harness and never
touches `variable_duration` / `_get_epoch_ix_range` / `_n_times_per_epoch`, so it runs
under the pre-PR worktrees. It records full figure state after every gesture to JSON + an
npz sidecar, then diffs two recordings. Arrays compare by **sha256 of the raw float64
bytes** — stricter than `array_equal`; it catches `-0.0` and NaN-payload differences too.
`parity_analyze.py` aggregates a diff by field with magnitudes.

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg PYTHONPATH="D:/tmp/mne-base;D:/tmp/qtb-base/src" \
  python parity.py record --backend qt --out D:/tmp/parity/base
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg PYTHONPATH="D:/mne-python;D:/mne-qt-browser/src" \
  python parity.py record --backend qt --out D:/tmp/parity/branch
python parity.py diff D:/tmp/parity/base D:/tmp/parity/branch --backends qt --summary
```

Both environments asserted per run; every recording carries its `mne.__file__`.

| run | backend | figures | states |
|---|---|---|---|
| epochs + raw/ICA + precompute | matplotlib | 94 | 2142 |
| epochs + raw/ICA + precompute | qt | 97 | 1993 |
| epochs, re-run with Qt bad-epoch clicks | qt | 90 | 2154 |
| broadened raw/ICA matrix | matplotlib | 48 | 712 |
| broadened raw/ICA matrix | qt | 48 | 696 |
| **total** | | **377 per environment** | **7697** |

Epoch specs: 1/2/4/5/50 epochs; lengths 1, 2, 40, 77, 100, 120, 137, 200, 250; 1/3/64
channels; sfreq 100/250/512.3/1000; tmin 0/−0.2/+0.1; three drop patterns giving
non-contiguous `selection`. Plot kwargs: `n_epochs` ∈ {1, 2, len, len+5}, `decim`
auto/1/2/4, `events=True`, `butterfly=True`, `scalings` auto and dict.

Gestures were verified non-vacuous: 1170/1206 matplotlib states carry a non-empty
`bad_epochs`, ~1000 have butterfly on, 800–1196 have a vline shown. The Qt script
initially never marked an epoch bad — a `qt_bad:<n>` action driving `DataTrace.toggle_bad`
(the exact path a left click takes) was added and the matrix re-run.

## What is bit-identical

Across all 7,697 shared states, **zero** divergence in `start`, `stop`, `loaded_data`,
`loaded_times`, `mne.data`, `mne.times`, `boundary_times`, `midpoints`, `n_times`,
`n_epochs`, `bad_epochs`, `info["bads"]`, `decim`, `scale_factor`, `butterfly`,
`n_channels`, `ch_start`, `first_time`, and every trace's x and y data. Qt additionally:
`viewRange()`, `epoch_idx`, `epoch_dur`, `xmax`, vline positions/visibility/labels,
scrollbar state, `_get_x_from_norm` at 7 positions, `epoch_color_ref`, overview-bar
bad-epoch rectangles. matplotlib additionally: `xlim`, `ylim`, `vline_visible`,
`n_epoch_traces`.

**`boundary_times` is not merely equal to 1e-12 — it is the same doubles.**
`np.arange(n+1) * L` (base) and `np.cumsum(np.full(n, L))` (branch) are both exact
integers in float64, so dividing by `sfreq` gives identical bit patterns. Checked
exhaustively over sfreq ∈ {10, 100, 250, 512.3, 1000, 2048, 44100} × L ∈ {1…1000} × n ∈
{1…20000}: identical everywhere. **The last-bit `searchsorted` concern raised in the plan
does not materialise.** The same sweep shows the derived
`boundary_samples = round(boundary_times * sfreq)` (the ICA path) round-trips exactly.

---

## P1 — matplotlib: clicking in the last half-sample of an epoch deletes the marker and labels it with a latency the epoch does not have

**Verdict: new defect, and it hits the equal-duration path. Base does not do this.**
Independently re-verified by the coordinator.

`_recompute_epochs_vlines` ([mne/viz/_mpl_figure.py:2445](D:/mne-python/mne/viz/_mpl_figure.py:2445))
computes an unclamped sample offset:

```python
offset = round((xdata - boundary_times[clicked_ix]) * sfreq)      # 2445
latency = self.mne.epoch_tmins[clicked_ix] + offset / sfreq       # 2446
...
if tmin - 0.5 / sfreq <= latency <= tmax + 0.5 / sfreq:           # 2451
```

`offset` can reach `L` (the epoch's sample count) for any click in the final half sample,
giving `latency = tmin + L/sfreq` — exactly `1/sfreq` beyond `tmax = tmin + (L-1)/sfreq`,
so it fails the keep test for *every* visible epoch. `xs` comes back empty,
`set_segments` gets an empty array, and `_show_vline` still writes the out-of-range
latency into `vline_text` and calls `_toggle_vline(True)`. The reader gets a latency
readout with no line under it.

4 equal-duration epochs, `n_epochs=2`, click at `frac` into epoch 0:

| L, sfreq, tmin | frac | click x | base lines / label | branch lines / label |
|---|---|---|---|---|
| 100, 100, 0 | 0.990 | 0.990 | 2 / `0.99 s` | 2 / `0.99 s` |
| 100, 100, 0 | 0.995 | 0.995 | 2 / `0.995 s` | **0** / `1 s` |
| 100, 100, 0 | 0.999 | 0.999 | 2 / `0.999 s` | **0** / `1 s` |
| 2, 100, 0 | 0.900 | 0.018 | 2 / `0.018 s` | **0** / `0.02 s` |
| 100, 512.3, −0.2 | 0.999 | 0.195003 | 2 / `-0.004 s` | **0** / `-0.004 s` |
| 200, 1000, 0.1 | 0.999 | 0.199800 | 2 / `0.2998 s` | **0** / `0.3 s` |

The real last sample is `0.99 s` (L=100) and `0.01 s` (L=2), so the branch labels `1 s`
and `0.02 s` — latencies that do not exist in the data.

Dead-zone width is `0.5/sfreq` seconds, i.e. `0.5/L` of the epoch: 0.5 % at L=100 (~2 px
in a typical window) but **25 % of every epoch at L=2**.

**Root cause, and why Qt is unaffected.** `MNEQtBrowser._get_vline_times` keeps the old
`t % epoch_dur` code behind `if is_variable_duration(self.mne)`, so its fixed path is
byte-for-byte preserved. The matplotlib rewrite has **no such guard** — the fixed path
goes through the new code. That asymmetry is the root of both P1 and P2.

Fix — clamp the offset to the clicked epoch's own sample count:

```python
n_samp = int(round((boundary_times[clicked_ix + 1] - boundary_times[clicked_ix]) * sfreq))
offset = int(np.clip(round((xdata - boundary_times[clicked_ix]) * sfreq), 0, n_samp - 1))
```

Verified by monkeypatch: all four configurations then draw 2 lines at every frac from 0.9
to 0.9999 and label the epoch's real last sample. Because the clamp uses the *clicked*
epoch's length, ragged behaviour is preserved — longer epochs still receive a line.

---

## P2 — matplotlib: the vline latency is now quantised to the nearest sample

**Verdict: fixed-path behaviour change. Arguably a fix, but visible, and should be stated
in the PR.**

Base reported the raw click position (`xdata % epoch_dur`); the branch snaps to
`round(offset * sfreq) / sfreq`. Every distinct label change in the 2142-state matrix:

| base | branch | spec | states |
|---|---|---|---|
| `0.018 s` | `0.02 s` | L=2, sfreq 100 | 36 |
| `0.693 s` | `0.69 s` | L=77, sfreq 100 | 36 |
| `0.006 s` | `0.01 s` | L=2 | 4 |
| `0.231 s` | `0.23 s` | L=77 | 4 |

Every branch value is on the sample grid; no base value is. The drawn line moves by at
most half a sample (max `|Δx|` = 0.004 s). Defensible — there is no datum at 0.231 s — but
at L=2 any click in `[0.005, 0.015)` reports `0.01 s`.

---

## P3 — the "two time points" guard moved from per-epoch to total, admitting 1-sample epochs; matplotlib then crashes at `n_epochs=1`

**Verdict: fixed-path behaviour change; new loud failure where base gave a clear message.**

Base `_figure.py:80` guards `len(inst.times) < 2`. Branch `_figure.py:114-121` guards
`self.mne.n_times < 2`, where for epochs `n_times` is the **total** concatenated count.
Two 1-sample epochs total 2 samples, so they now pass.

| | base (both backends) | branch matplotlib | branch qt |
|---|---|---|---|
| `n_epochs=1` | `ValueError: Data from at least two time points are required…` | **`IndexError: index 0 is out of bounds for axis 0 with size 0`** | opens |
| `n_epochs=2` | same `ValueError` | opens | opens |

The crash is at `_mpl_figure.py:2268`, `epoch_nums = self.mne.inst.selection[epoch_ix[0] : epoch_ix[-1] + 1]`
— **byte-identical to base**, only newly reachable. With `n_epochs=1` over a 1-sample
epoch, `mne.times` has length 1, so `time_range = times[[0, -1]]` is degenerate,
`searchsorted` returns `[0, 0]`, and `epoch_ix = np.arange(0, 0)` is empty. Each such
epoch also has `tmin == tmax` (this resolves seed suspect **S10**).

Two options: restore a per-epoch minimum for the fixed path, or make `_draw_traces`
tolerate an empty `epoch_ix`. Qt drives all 24 gestures on this figure without error, so
only matplotlib needs the second guard.

---

## P4 — `sampling_period` differs by 1 ULP whenever `tmin != 0`; provably inert

**Verdict: benign.** This resolves seed suspect **S4**.

Base: `np.diff(inst.times[:2])[0] / sfreq`. Branch: `(1.0/sfreq)/sfreq`. When
`start_idx == round(tmin*sfreq)` is 0 these are the same double; otherwise they differ by
one ULP (relative ~8.5e-16).

It appeared in 888 (mpl) / 900 (qt) states — every state of every `tmin != 0` spec — and
in **no** downstream field. It is used at exactly one site as a nudge before
`np.searchsorted(boundary_times, t_start - sampling_period)`. Swept **717,920**
(sfreq, L, tmin, n, boundary) combinations over sfreq ∈ {10…44100}, L ∈ {1…1000},
tmin ∈ {0, ±0.2, ±1.5, 3}, n up to 2000: **zero searchsorted flips**. The margin is
structural — `t_start - sp` sits at least `L*sfreq - 1` nudge-widths above the boundary
below it (minimum 9× at L=1, sfreq=10), while the two nudges differ by 8.5e-16 of a nudge.

Raw and ICA are unaffected: `RawArray.times` is `arange(n)/sfreq` so `diff(times[:2])` is
exactly `1/sfreq`; and `ica.plot_sources` builds its source array with default `tmin=0`.

---

## P5 — sub-femtosecond ULP drift in `t_start`, `duration`, patch and segment geometry

**Verdict: benign; the branch is the cleaner of the two.** matplotlib only. `t_start` max
`|Δ|` 1.44e-15 s, `duration` 8.9e-16 s, hscroll patch geometry 2.1e-15 axes units. The
branch always lands on the nearest double (0.4 = `0x1.999999999999ap-2`) while base
accumulates up to 6 ULP from repeated scrolling; at `home`, base leaves a residue of
`2^-51` where the branch returns exactly `0.0`. `start`, `stop`, `xlim`, `loaded_data`,
`times` and all traces are identical in exactly those states.

---

## Raw and ICA

**No divergence at all on Qt** (696 states, 48 figures). On matplotlib (712 states) the
only divergences are the P5 ULP class confined to the ICA-epochs group, plus P2. Covered:
`raw.plot` at sfreq 100/250/512.3/1000 with `first_samp` 0/733/1001 and 1/6/64 channels,
across `duration=3`, `duration=1e6`, butterfly, `decim` 2 and 4, and a scalings dict;
`ica.plot_sources(raw)`; `ica.plot_sources(epochs)`; and `ICA.fit(epochs)` followed by
`plot_sources(epochs)` at 4 sfreq/tmin/length combinations.

The two `BrowserBase.__init__` changes both behave: the derived
`boundary_samples`/`epoch_tmins`/`epoch_tmaxs` fallback is entered only on the ICA-epochs
path and reproduces the caller-supplied values exactly there; the `sampling_period`
redefinition is P4.

`precompute=True` on Qt (3 specs including sfreq 512.3 and a non-contiguous `selection`):
`global_times` and every captured field identical.

---

## Ranked summary

| # | severity | finding | backend | verdict |
|---|---|---|---|---|
| P1 | silent wrong picture | vline vanishes and mislabels for clicks in an epoch's last half sample | matplotlib | **new defect on the fixed path**; fix verified |
| P2 | visible change | vline latency quantised to nearest sample | matplotlib | fixed-path change; probably intended, document it |
| P3 | loud exception | 1-sample epochs admitted; `n_epochs=1` raises `IndexError` not a clear `ValueError` | matplotlib | **new on the fixed path** |
| P4 | none | `sampling_period` 1 ULP when `tmin != 0` | both | inert; proven over 717,920 combinations |
| P5 | none | ULP drift in `t_start`/`duration`/geometry | matplotlib | inert; branch more accurate |

On the 32 fields that determine what the reader actually sees, parity is exact across
7,697 states and 377 figures per environment.
