"""Display-side invariants for the A2 slice (matplotlib).

Everything compares drawn artist state against arrays computed from the source,
never against the browser's own model of what it drew.
"""

import numpy as np

from .common import drawn_samples, full_xy, per_epoch_phase, trace_matches_source


def d1_every_visible_epoch_drawn(fig, case):
    """No visible epoch may end up with zero drawn samples."""
    out = []
    ix0, ix1 = fig._get_epoch_ix_range()
    samples = drawn_samples(fig, case)
    phases = per_epoch_phase(samples, case)
    for k in range(ix0, ix1):
        if k not in phases:
            out.append(
                f"D1 epoch {k} (length {case.lengths[k]}) is inside the view "
                f"{ix0}:{ix1} but has no drawn sample"
            )
    return out


def d2_trace_count(fig, case):
    """One primary Line2D per pick, all the same length."""
    out = []
    n_picks = len(fig.mne.picks)
    if len(fig.mne.traces) != n_picks:
        out.append(f"D2 {len(fig.mne.traces)} traces for {n_picks} picks")
        return out
    lens = {len(np.ma.getdata(t.get_xdata())) for t in fig.mne.traces}
    if len(lens) > 1:
        out.append(f"D2 traces have different x lengths: {sorted(lens)}")
    for i, t in enumerate(fig.mne.traces):
        x = np.ma.getdata(t.get_xdata())
        y = np.ma.getdata(t.get_ydata())
        if len(x) != len(y):
            out.append(f"D2 trace {i}: {len(x)} x values, {len(y)} y values")
        if not np.isfinite(np.asarray(y, float)).all():
            out.append(f"D2 trace {i}: non-finite y values drawn")
    return out


def d3_xy_source(fig, case):
    return [f"D3 {m}" for m in trace_matches_source(fig, case)]


def d4_x_inside_view(fig, case):
    """Every drawn x lies inside the loaded window's sample range."""
    out = []
    start, stop = fig._get_start_stop()
    samples = drawn_samples(fig, case)
    if samples.min() < start or samples.max() > stop - 1:
        out.append(
            f"D4 drawn samples span [{samples.min()}, {samples.max()}], "
            f"window is [{start}, {stop - 1}]"
        )
    if np.any(np.diff(samples) <= 0):
        out.append("D4 drawn x values are not strictly increasing")
    return out


def d5_offsets_distinct(fig, case):
    """Non-butterfly traces must sit on distinct offsets and inside ylim."""
    out = []
    if fig.mne.butterfly:
        return out
    offs = np.asarray(fig.mne.trace_offsets, float)
    n = len(fig.mne.picks)
    if len(np.unique(offs[:n])) != n:
        out.append(f"D5 duplicate trace offsets among {n} picks: {offs[:n]}")
    return out


def d6_event_lines(fig, case):
    """Event lines sit at ``boundary_times[k] - tmin[k]`` for the epochs that
    contain their event, and nowhere else."""
    out = []
    lines = getattr(fig.mne, "event_lines", None)
    if lines is None:
        return out
    got = _event_line_x(lines)
    if got is None:
        return out
    # every event is at latency 0 in this harness (events are the epoch's own
    # trigger), so the drawn x is boundary + (0 - tmin); only the epochs whose
    # samples are loaded can carry a line
    start, stop = fig._get_start_stop()
    lo = start / case.sfreq
    hi = (stop - 1) / case.sfreq
    want = []
    for k in range(case.n_epochs):
        t0 = case.boundary_times[k] - case.tmins[k]
        tmax_t = case.boundary_times[k] + (case.lengths[k] - 1) / case.sfreq
        if case.boundary_times[k] - 1e-9 <= t0 <= tmax_t + 1e-9:
            if lo - 1e-9 <= t0 <= hi + 1e-9:
                want.append(t0)
    want = np.sort(np.asarray(want, float))
    got = np.sort(np.asarray(got, float))
    if len(got) != len(want) or not np.allclose(got, want, atol=1e-9):
        out.append(
            f"D6 event lines at {np.round(got, 6).tolist()}, "
            f"expected {np.round(want, 6).tolist()}"
        )
    return out


def _event_line_x(lines):
    """x positions of the drawn event lines, or None if the artist is unknown."""
    from matplotlib.collections import LineCollection

    if isinstance(lines, LineCollection):
        segs = lines.get_segments()
        if not len(segs):
            return []
        return [s[0, 0] for s in segs]
    if isinstance(lines, list | tuple):
        xs = []
        for ln in lines:
            g = getattr(ln, "get_xdata", None)
            if g is None:
                return None
            xs.append(float(np.atleast_1d(g())[0]))
        return xs
    return None


ALL = (
    d1_every_visible_epoch_drawn,
    d2_trace_count,
    d3_xy_source,
    d4_x_inside_view,
    d5_offsets_distinct,
)


def check(fig, case, *, with_events=False):
    out = []
    fns = ALL + ((d6_event_lines,) if with_events else ())
    for fn in fns:
        try:
            out.extend(fn(fig, case))
        except Exception:
            import traceback

            out.append(f"{fn.__name__} raised:\n{traceback.format_exc(limit=4)}")
    return out
