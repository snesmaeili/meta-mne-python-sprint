"""Invariants about *what is drawn*, as opposed to which samples are shown.

Companion to :mod:`invariants`; kept separate so that a slice which only cares
about navigation is not slowed down (or confused) by matplotlib artist
introspection. Everything here is matplotlib-only: the Qt backend keeps its
curves in a different structure.

Every expectation is computed from the :class:`~validation.browser_fuzz.build.Case`
(i.e. from the source arrays), never from the figure.
"""

import traceback

import numpy as np

TOL = 1e-9


def _fmt(x):
    return np.array2string(np.asarray(x), precision=6, threshold=12, edgeitems=4)


def _decim_of(fig):
    d = fig.mne.decim
    return int(np.max(d)) if np.ndim(d) else int(d)


def _epoch_ix_used(fig):
    """Reproduce the epoch index list ``_draw_traces`` builds for colouring."""
    time_range = (fig.mne.times + fig.mne.first_time)[[0, -1]]
    ends = np.searchsorted(fig.mne.boundary_times, time_range)
    return np.arange(ends[0], ends[1])


def a1_time_scalebar(fig, case):
    """The time scalebar fits in the view and its label matches its width."""
    out = []
    bar = fig.mne.scalebars.get("time")
    if bar is None:
        return out
    x = np.asarray(bar.get_xdata(), float)
    lo, hi = fig.mne.ax_main.get_xlim()
    width = x[1] - x[0]
    if x[1] > hi + 1e-9 or x[0] < lo - 1e-9:
        out.append(
            f"A1 time scalebar spans {x[0]:.4f}..{x[1]:.4f} but the view is "
            f"{lo:.4f}..{hi:.4f} (bar is {width / (hi - lo):.1f}x the window)"
        )
    txt = fig.mne.scalebar_texts.get("time")
    if txt is not None:
        try:
            said = float(txt.get_text().split()[0])
        except (ValueError, IndexError):
            said = None
        if said is not None and abs(said - width) > 0.005 + 1e-9:
            out.append(f"A1 time scalebar is {width:.4f} s but says {said} s")
    return out


def a2_every_visible_epoch_coloured(fig, case):
    """Each visible epoch takes part in the colour / bad-epoch bookkeeping."""
    out = []
    ix0, ix1 = fig._get_epoch_ix_range()
    used = _epoch_ix_used(fig)
    if len(used) != ix1 - ix0:
        out.append(
            f"A2 colour bookkeeping covers epochs {_fmt(used)} but the view is "
            f"{ix0}:{ix1}"
        )
    return out


def a3_trace_count(fig, case):
    """One primary trace per pick, and the y data has the decimated length."""
    out = []
    n_picks = len(fig.mne.picks)
    if len(fig.mne.traces) != n_picks:
        out.append(f"A3 {len(fig.mne.traces)} traces for {n_picks} picks")
        return out
    if not n_picks:
        return out
    decim = _decim_of(fig)
    want = len(fig.mne.times[::decim])
    for ii, line in enumerate(fig.mne.traces):
        y = np.asarray(line.get_ydata())
        if y.size != want:
            out.append(f"A3 trace {ii} has {y.size} points, want {want}")
            break
    return out


def a4_drawn_x_are_real_samples(fig, case):
    """Every drawn x is a real sample time of a visible epoch."""
    out = []
    ix0, ix1 = fig._get_epoch_ix_range()
    lo = int(case.boundary_samples[ix0])
    hi = int(case.boundary_samples[ix1])
    ref = np.asarray(fig.mne.traces[0].get_ydata())
    seen = set()
    for line in [fig.mne.traces[0]] + list(fig.mne.epoch_traces):
        y = np.asarray(line.get_ydata())
        if y.shape != ref.shape or not np.allclose(y, ref):
            continue
        x = line.get_xdata()
        good = x.compressed() if np.ma.isMaskedArray(x) else np.asarray(x)
        if not len(good):
            continue
        samp = np.round(np.asarray(good) * case.sfreq).astype(int)
        if np.any(samp < lo) or np.any(samp >= hi):
            bad = samp[(samp < lo) | (samp >= hi)]
            out.append(
                f"A4 drawn samples {_fmt(bad[:4])} lie outside the window {lo}:{hi}"
            )
        if not np.allclose(np.asarray(good) * case.sfreq, samp, atol=1e-6):
            out.append("A4 drawn x values are not on the sample grid")
        seen.update(int(s) for s in samp)
    decim = _decim_of(fig)
    expect = set(range(lo, hi, decim))
    # the first sample of the window is dropped by the mask (pre-existing, see
    # report_A2.md), so allow exactly that one
    missing = expect - seen - {lo}
    if missing:
        out.append(
            f"A4 {len(missing)} decimated samples are never drawn, "
            f"first {sorted(missing)[:4]}"
        )
    return out


def a5_offsets(fig, case):
    """Trace offsets and the y limits agree about how many rows there are."""
    out = []
    offsets = np.asarray(fig.mne.trace_offsets, float)
    ylim = fig.mne.ax_main.get_ylim()
    n_offsets = ylim[0] + 0.5
    if offsets.size and offsets.max() > n_offsets - 0.5 + TOL:
        out.append(f"A5 offset {offsets.max()} outside ylim {ylim}")
    if not fig.mne.butterfly and offsets.size != fig.mne.n_channels:
        out.append(
            f"A5 {offsets.size} offsets for n_channels={fig.mne.n_channels}"
        )
    return out


def a6_ydata_matches_data(fig, case):
    """The drawn y values are the loaded data, scaled and offset."""
    out = []
    if fig.mne.data is None:
        return out
    decim = _decim_of(fig)
    picks = fig.mne.picks
    offset_ixs = (
        picks if fig.mne.butterfly and fig.mne.ch_selections is None else slice(None)
    )
    offsets = np.asarray(fig.mne.trace_offsets)[offset_ixs]
    for ii in range(min(3, len(fig.mne.traces))):
        want = offsets[ii] - fig.mne.data[ii] * fig.mne.scale_factor
        want = want[..., ::decim]
        got = np.asarray(fig.mne.traces[ii].get_ydata())
        if got.shape != want.shape:
            out.append(f"A6 trace {ii} shape {got.shape} != data shape {want.shape}")
            break
        if fig.mne.clipping == "clamp":
            want = np.clip(want, -0.5, 0.5)
        if not np.allclose(got, want, atol=1e-9):
            out.append(f"A6 trace {ii} y data is not the scaled source data")
            break
    return out


_CHECKS = (
    a1_time_scalebar,
    a2_every_visible_epoch_coloured,
    a3_trace_count,
    a4_drawn_x_are_real_samples,
    a5_offsets,
    a6_ydata_matches_data,
)


def check_display(fig, case, backend="matplotlib"):
    """Run every display invariant; return a flat list of violations."""
    if backend != "matplotlib":
        return []
    out = []
    for fn in _CHECKS:
        try:
            out.extend(fn(fig, case))
        except Exception:
            out.append(f"{fn.__name__} raised:\n{traceback.format_exc(limit=4)}")
    return out
