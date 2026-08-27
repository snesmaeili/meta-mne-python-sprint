# A8 — Visual verification, both backends

Every panel below was rendered offscreen, saved as a PNG, **opened and looked at**, and
printed alongside the numbers behind it. Ragged data is compared against an equal-duration
control through the identical gesture. Representative data throughout: 20 EEG channels of
full-scale noise, not three clean traces. No repo file was modified; the one candidate fix
was validated by monkeypatch.

- Shots: `shots/` (99 files)
- Scripts: `a8_visual/` (`vis.py` + `s1`–`s10`, `compose.py`, `crop.py`)

```bash
QT_QPA_PLATFORM=offscreen QT_QPA_FONTDIR="C:/Windows/Fonts" MPLBACKEND=Agg \
  PYTHONPATH="D:/mne-python;D:/meta-mne-python-sprint" python a8_visual/s2_f1_vline.py
```

## M1 — methodology warning for anyone grabbing Qt screenshots

**`QT_QPA_FONTDIR` is mandatory.** Under bare `QT_QPA_PLATFORM=offscreen` on this machine
`QFontDatabase.families()` returns **0 families** and every string in a grab — channel
names, epoch numbers, vline labels, axis titles — renders as an empty box. The first pass
produced screenshots that looked like a working browser with no text anywhere. With
`QT_QPA_FONTDIR=C:/Windows/Fonts` there are 288 families and the text appears. Any earlier
"the screenshot looked fine" is worthless without this. Now in `BRIEF.md`.

---

## V1 — F1 on screen: the vline is glued to the short epoch's last sample, and its label disagrees with its sibling

**Verdict: new defect, ragged-only. This is the figure for the PR.**

**`shots/a8_FIGURE_f1_vline_clamped.png`** (4-panel composite). Components:
`a8_f1_ragged_1_before_scroll.png`, `a8_f1_ragged_2_after_scroll.png`,
`a8_mplvline_ragged_2_after_scroll.png`, `a8_f1_fixed_2_after_scroll.png`, plus
`a8_f1_ragged_4_permanent.png`.

Epochs 100/250/75/180 @100 Hz, boundaries `[0, 1, 3.5, 4.25, 6.05]`, `n_epochs=2`, one
`_add_vline(0.9)`, one `hscroll("right")`:

| | slot 0 | slot 1 |
|---|---|---|
| before scroll | x=0.900, epoch 0, latency **0.900**, label `0.900 s` | x=1.900, epoch 1, latency **0.900**, label `0.900 s` |
| after scroll | x=1.900, epoch 1, latency **0.900**, label `0.900 s` | x=4.240, epoch 2 (0.75 s long), latency **0.740**, label `0.740 s`, **visible** |
| fixed control | latency 0.900 | latency 0.900 |

On screen the second line sits **flush against the epoch boundary**, one pixel inside it,
and its label reads a different number from the first line's. Two lines, one gesture, two
latencies. Cropped labels side by side: `crop_f1_labels.png`.

**The matplotlib backend gets this right**, which is the strongest argument in the figure.
`_recompute_epochs_vlines` ([mne/viz/_mpl_figure.py:2424](D:/mne-python/mne/viz/_mpl_figure.py:2424))
computes the latency and emits a segment only for epochs whose `[tmin, tmax]` reaches it.
After the same scroll it draws **one** segment (epoch 1 at 0.900 s) and none in epoch 2.
Qt's `_xrange_changed` `bmin + rel_vl_t` loop should do what matplotlib already does.

Permanence confirmed visually (`a8_f1_ragged_4_permanent.png`): right, right, left, left
ends at `[0.740, 0.740]` — 0.900 gone from both lines. Fixed control holds `[0.900, 0.900]`
at every step.

---

## V2 — the overview-bar click paints a quarter of the window blank white

**Verdict: new defect, ragged-only.** Pictorial confirmation of A4-1, plus one new
consequence.

**`shots/a8_FIGURE_overview_click_blank.png`**. Components: `a8_ovclick_ragged_0_open.png`,
`a8_ovclick_ragged_1_click_far_right.png`, `a8_ovclick_fixed_1_click_far_right.png`.

One `QTest.mouseClick` at the right end of the overview bar, epochs 100/250/75/180,
`n_epochs=2`:

```
view      = [2.5500, 6.0500]   width 3.500 s
epochs    = 2:4   loaded data spans [3.5000, 6.0400]
axis says = ['2', '3']
blank     = 0.960 s = 27.4% of the window
both ends on a boundary: False
```

The **left 27% of the plot area is empty white** — no traces, no boundary line — while the
epoch axis underneath reads `2  3`. Fixed control through the identical click gives
`[3.000, 6.000]`, epochs 2:4, nothing blank.

**New, not in A4-1:** at `n_epochs=1` the same gesture (click 25% along the bar) gives view
`[1.000, 2.000]` inside epoch 1, which spans `[1.000, 3.490]`. `TimeAxis.tickValues`
selects midpoints inside the view; epoch 1's midpoint (2.25) is outside it, so
`get_labels()` returns `[]` and **the axis carries no epoch number at all** — a window
showing 40% of one epoch, unlabelled, with no separator in sight
(`a8_ovclick_ragged_3_one_epoch_25pct.png`). Fixed control: view `[0.000, 1.500]`, label
`['0']`.

---

## V3 — epoch numbers vanish from the Qt axis while `get_labels()` still reports them

**Verdict: pre-existing mechanism, newly reachable in ordinary configurations on the ragged
path.** Recorded in `FINDINGS.md`. Initially scored ragged-only; the 50- and 100-epoch
fixed controls corrected that.

**`shots/a8_FIGURE_missing_epoch_labels.png`**. Full table and the validated candidate fix
are in `FINDINGS.md` under V3.

---

## Checked and visually correct

Each rendered ragged *and* equal-duration, looked at, and cross-checked against numbers
computed from the source arrays.

- **B1 Boundary separator lines, both backends.** Qt draws separators at exactly
  `boundary_times[1:-1]`, in view and nowhere else, at 300/420/600/900/1600 px, for
  `(3,300,3,300)`, `(3,3,300,300)`, `(2,3,2,3)`, `(100,250,75,180)` and the equal control.
  Pixel positions matched `width * t / xmax` to two decimals throughout. matplotlib's
  `FixedLocator(boundary_times[1:-1])` gridlines likewise. A 3-sample epoch beside a
  300-sample one renders as a ~4 px sliver bounded by two lines that merge into one thick
  bar — cramped, but drawing exactly what the data says.
- **B2 matplotlib horizontal scrollbar, 50 epochs.** One `Rectangle` per epoch, `x` and
  `width` errors both exactly `0.000e+00` against `boundary_times`. Pixel widths span
  2.53–27.68 px (median 15.04), **none sub-pixel**, all legible. Widths visibly track
  durations. All 17 scrollbar labels name the right epoch, on ragged and equal alike.
  `crop_scroll_ragged50.png` vs `crop_scroll_equal50.png`.
- **B3 Epoch numbers under the right epochs when widths differ.** matplotlib main axis,
  `(100,250,75,180)`: labels at 62 / 279 / 480 / 638 px — unevenly spaced, each at its own
  midpoint, all correct.
- **B4 Dropped epochs.** `drop=(1,3)` from five: both backends label the survivors `0 2 4`
  at midpoints `[0.5, 1.375, 2.35]` (ragged) and `[0.75, 2.25, 3.75]` (fixed).
- **B5 Event lines with per-epoch `tmin`.** `tmin=[0, -0.2, 0.1, -0.5]`, lengths
  `(100,250,75,180)`: `EventLine` x = `[0.0, 1.2, 4.75]` — exactly
  `boundary_times[i] - tmin[i]` for the three epochs containing t=0, and **no line for
  epoch 2**, whose `tmin=+0.1` puts the event before its first sample. On screen the three
  lines sit at visibly different offsets from their boundaries, which is the point of
  per-epoch `tmin`.
- **B6 Butterfly.** Ragged separators at `1.0/3.5/4.25` → 138.7 / 485.4 / 589.4 px of 839;
  fixed → 209.8 / 419.5 / 629.2. All four epoch numbers painted in both.
- **B7 Resize.** 900×450 → 300×200 → 1500×700 → back to 900×450, ragged and fixed:
  separator pixel positions matched the recomputed expectation to **two decimals at every
  size**, including on return. Nothing pixel-positioned goes stale across a resize on this
  branch. (Independently agrees with A4's numeric resize probe.)
- **B8 Themes — what was actually rendered.** `theme="dark"` did flip `mne.dark` to
  `True` under offscreen here: widget background `(0.098, 0.137, 0.176)`, ragged and fixed.
  A forced Fusion dark palette gave the same result, so these shots are genuinely dark and
  the ragged dark rendering is correct: separators visible, all four epoch numbers painted.

  **Correction (coordinator):** this does *not* contradict the AGENTS.md caution, as this
  report originally claimed. `_qt_get_stylesheet` (`mne/viz/backends/_utils.py:388`)
  returns an **empty stylesheet** — a silent no-op — when `theme == "auto"`, when
  `theme == system_theme`, or when qdarkstyle is missing (which only `logger.info`s). This
  machine's OS theme is *light* and qdarkstyle 3.2.3 is installed, so `theme="dark"`
  differed from the system and produced a 54 KB stylesheet. On a machine whose OS theme is
  already dark, `theme="dark"` matches the system, the stylesheet is empty, and the widget
  inherits the platform palette — which under offscreen/xvfb does not exist, so it falls
  back to light and `mne.dark` is `False`. Eric's warning holds; it just bites on a
  dark-themed host rather than because of offscreen as such. **Assert `fig.mne.dark`
  before trusting any dark screenshot.**
  - Cosmetic, **pre-existing**: the `"Epoch Index"` axis title is clipped at the bottom
    edge in dark — identical on the fixed path (`crop_dark_fixed_axis.png`), absent in
    light on both. Out of scope.
  - The `Loading… 100%` progress bar persisting in the status bar of several Qt grabs
    appears identically on the fixed path. Not a finding.

## Coordinator notes

- The three `a8_FIGURE_*.png` composites are captioned and ready to attach to the PR
  discussion as-is.
- V3's self-correction matters: reported ragged-only until the fixed controls came back
  dirty.
- V2's `n_epochs=1` variant has been added to `report_A4.md` under A4-1.
- M1 has been added to `BRIEF.md`.
