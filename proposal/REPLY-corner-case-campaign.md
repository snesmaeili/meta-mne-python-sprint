# Reply — corner-case campaign (mne-python #14210 / mne-qt-browser #452)

> Draft. Two audiences in one comment: @drammock asked for the corner-case hunt,
> @larsoner asked for the AGENTS.md adjustments and a smaller diff.

---

Done — and it found more than I expected, including three things that break
**equal-duration** browsing, which matters more than anything ragged-specific.

## First, a blocker I had to clear

Every Qt browser test in mne-tools/mne-qt-browser#452 was skipping on any real install, for
two independent reasons: `mne` resolves to the released 1.11.0 unless you happen to be
sitting in the checkout, and `plot_epochs` gated on the backend *name* before
`_get_browser` was reached. So that half of the work had never actually run.

Replaced the name gate with a capability check, duck-typed the way `BrowserBase._has_time_slice`
already does it — `_pg_figure` advertises `_SUPPORTS_VARIABLE_DURATION`, an older
mne-qt-browser simply doesn't, and declines with the same message as before. Qt went from
0 passed / 7 skipped to **9 passed / 0 skipped**.

While there: six tests on the MNE side carried the skip reason *"variable-duration browsing
is matplotlib-only"*, which was no longer true. Qt matches matplotlib exactly on arrow
keys, shift-arrows, home/end and all the window arithmetic, so those now run on both
backends (17 passed). Only the four that assert on matplotlib artefacts still skip, with
an honest reason.

## What the campaign was

A harness plus eight agent slices, all headless. The rule throughout: **anything that also
happens with equal-duration epochs is pre-existing and out of scope**, checked against
`force_fixed` and, where it still looked new, against `c4f5ba1e9` in a worktree.

- 2,640 randomised runs (44 specs × 30 seeds × 40 steps, both backends) with normalised
  failure signatures and a delta-debugging shrinker
- 7,697 recorded states diffed byte-for-byte against the pre-PR commits
- ~650 plot-argument × channel matrix cells
- 99 screenshots, opened and looked at

The fuzzer has a mutation self-test — six injected defects, six caught, a 40-step failure
shrinking to 2 steps. A clean fuzz result is worthless without that, and I'd rather say so
than present the number bare.

## The part that matters most: three regressions on the fixed path

These are not ragged-only. They change browsing that works on `main` today.

**1. A click in an epoch's last half-sample deletes the vline and mislabels it.**
`_recompute_epochs_vlines` computes an unclamped sample offset, which can round up to one
sample past the epoch's end — a latency no epoch holds — so *every* line is dropped while
the readout still shows the out-of-range value. Four ordinary 100-sample epochs:

| click x | `main` | this branch (before fix) |
|---|---|---|
| 0.990 | 2 lines, `0.99 s` | 2 lines, `0.99 s` |
| 0.995 | 2 lines, `0.995 s` | **0 lines**, `1 s` |

The dead zone is `0.5/L` of each epoch — 0.5 % at 100 samples, **25 % at 2**. Root cause is
an asymmetry: the Qt backend kept its old `t % epoch_dur` path behind an
`is_variable_duration` guard, the matplotlib rewrite has no such guard, so fixed-duration
data goes through the new code. Fixed by clamping the offset into the clicked epoch.

**2. 1-sample epochs are now admitted and then crash.** The "at least two time points"
guard moved from per-epoch to the *total* concatenated count, so two 1-sample epochs pass
it. `main` raised a clear `ValueError`; the branch raised
`IndexError: index 0 is out of bounds for axis 0 with size 0`.

**3. The vline latency is now quantised to the sample grid** (`0.231 s` → `0.23 s`).
Probably intended — there is no datum at 0.231 s — but it is a visible change and belongs
in the description rather than being discovered.

## Ragged-only defects, fixed

Six silent-wrong-picture, five loud. The ones worth naming:

- **Qt vlines drifted permanently.** `_xrange_changed` repositioned them by a raw
  seconds-offset instead of converting through `latency_at`, and never re-applied
  visibility. A line clamped into a short epoch then became the *next* scroll's reference,
  so scrolling right twice and back left both lines at 0.740 s with the original 0.900 s
  gone. The tightest repro needs no unequal durations at all — four equal-length epochs
  with per-event `tmin` are enough, which is worth noting since it means the defect class
  is broader than "different durations".
- **An overview-bar click showed a window that wasn't the epochs it claimed** — up to 27 %
  of the plot blank white while the axis still read `2 3`. Both it and the time scrollbar
  picked a start epoch and then reused the current `mne.duration` as the width; with ragged
  epochs the duration of *n* epochs depends on *which* *n*.
- **Subsetting left the time axis at the parent's span.** `_getitem` moved the per-epoch
  bounds correctly but never re-derived `_raw_times`, so after `epochs[0]` a single
  100-sample epoch produced `as_fixed()` of shape `(1, 3, 280)` with 540 NaN and
  `n_contributing == 0` at 180 time points — the very array that exists so a reader can see
  how many epochs back each point. `crop()` re-derives it; `_getitem` didn't. The codebase
  disagreeing with itself is what settles this as an omission rather than a choice.
- **`h` (epoch histogram) crashed on both backends** — `np.ptp(get_data(), axis=2)` on a
  list. Now computed per epoch.
- **A 1-sample epoch was never drawn as itself** — marking it bad changed nothing on screen
  (0 red pixels against 537 in the control), and marking its *neighbour* painted it red
  instead.

## Answering your specific questions

**@larsoner, the pending AGENTS.md** (mne-tools/mne-qt-browser#450): applied. Its "look at
it before you believe it" section changed the campaign — I added a slice that renders and
actually opens the images, and it caught things the assertions did not. One thing worth
folding into that file: under bare `QT_QPA_PLATFORM=offscreen` on Windows,
`QFontDatabase.families()` returns **0** and every string in a `grab()` renders as an empty
box, so a screenshot can look like a working browser with no text anywhere.
`QT_QPA_FONTDIR` is required. Its `PYTHONPATH=$PWD/src` note is also exactly right — I'd
derived the same workaround the hard way before reading it.

Also checked what it warns about: nothing ragged-specific is pixel-positioned (resize is
clean at five window sizes down to 2-px epochs), and nothing new is stored on `self.mne`
that `closeEvent` would need to drop — the branch adds only arrays and a float.

Your colour caution is right, and worth stating more sharply than "offscreen" in that file.
`_qt_get_stylesheet` returns an **empty stylesheet** — a silent no-op — in three cases:
`theme="auto"`, `theme == system_theme`, or qdarkstyle missing (which only `logger.info`s,
so nothing surfaces). So on a host whose OS theme is already dark, `theme="dark"` matches
the system, the stylesheet is empty, and the widget inherits a platform palette that under
offscreen/xvfb does not exist — you get a light render you believe is dark. On a
*light*-themed host with qdarkstyle installed it works, which is how one of my slices
initially concluded the caution was wrong. The robust instruction is to assert
`fig.mne.dark` before trusting a dark screenshot, rather than to trust the platform.

**@larsoner, the diff size.** Stripping private-helper docstrings to their first line is
worth **~199 lines** in mne-python across 13 helpers new in this branch, and ~99 in
mne-qt-browser. Three helpers are excluded because they predate the PR, where removing a
docstring would *grow* the diff. Happy to do it — I held off only because it overlaps the
simplification pass below and I'd rather touch those lines once.

**@drammock, simplification candidates.** Three near-identical decorator factories that
differ only in message text; `_VARIABLE_FALLBACK` holding one entry plus a `note` mechanism
nothing uses; `_epoch_window` duplicated across both repos; and `sampling_period`, whose own
comment concedes it isn't a sampling period — I swept 717,920 combinations and it never
changes a `searchsorted` outcome, so it can be renamed honestly or replaced with an explicit
`side="right"`.

Also: **`epoch_dur` in mne-qt-browser is dead weight.** Its only two readers are behind
`is_variable_duration`, and it's set from the *first* epoch, so it is wrong whenever it
could matter and unreachable whenever it is wrong.

## Three decisions I'd rather you made

1. **Should 1-sample epochs be admitted at all?** That single call resolves regression 2
   and all three symptoms of the 1-sample drawing defect. I fixed the draw site, which is
   needed either way, and left the guard alone.
2. **Should `_getitem` re-collapse `_variable_duration` the way `crop()` does?** Right now
   selecting two equal-length epochs out of a ragged set leaves an object that raises
   `These 2 epochs have durations from 0.990 to 0.990 s, so there is no time axis they
   share.`
3. **`concatenate_epochs` cannot be classified.** It's a module-level function, so the
   three `setattr` loops can't reach it however the dicts are edited. That's a hole in the
   frozen-list design rather than a missing entry. `drop_bad(reject=…)` was the same shape
   and now raises a clear message instead of an internal `RuntimeError`; note a blanket
   entry there would have broken the no-arg `drop_bad()` that `_concatenate_epochs` calls
   internally.

## Deliberately not fixed

Six behaviours reproduce with equal-duration epochs and are recorded rather than touched —
`_xtick_formatter` raising with fewer than two ticks, the Qt vline list not resizing on the
fixed path, epoch labels dropped by pyqtgraph when their text rect overflows the axis, and
the decim phase drifting across boundaries (which `main` does too, byte-identically).

Two are worth one line in the description because the PR silently *changes* them: the
ragged branch handles event bounds at the extremes **correctly** where the fixed branch does
not, so the two paths now disagree; and `boundary_times` is not merely close to the old
`n*len/sfreq` but the same doubles, so there is no float-drift concern to worry about.

Harness, per-slice reports and screenshots are in the sprint repo if any of this is worth
reproducing.
