"""Invariants the variable-duration browser must hold after every action.

Each check returns a list of strings; empty means it held. Nothing here raises
on a violation, so a fuzz run can keep walking and report everything it saw.
"""

import traceback

import numpy as np

TOL = 1e-9


def _fmt(x):
    return np.array2string(np.asarray(x), precision=6, threshold=12, edgeitems=4)


def i0_model(fig, case):
    """The boundary model matches the source arrays."""
    out = []
    bt = np.asarray(fig.mne.boundary_times, float)
    bs = np.asarray(fig.mne.boundary_samples, int)
    if len(bt) != case.n_epochs + 1:
        out.append(f"I0 boundary_times has {len(bt)} entries, want {case.n_epochs + 1}")
        return out
    if not np.allclose(bt, case.boundary_times, atol=TOL):
        out.append(f"I0 boundary_times {_fmt(bt)} != {_fmt(case.boundary_times)}")
    if not np.array_equal(bs, case.boundary_samples):
        out.append(f"I0 boundary_samples {_fmt(bs)} != {_fmt(case.boundary_samples)}")
    if int(fig.mne.n_times) != int(case.lengths.sum()):
        out.append(f"I0 n_times {fig.mne.n_times} != {case.lengths.sum()}")
    return out


def i1_window_integrity(fig, case):
    """_get_start_stop and _get_epoch_ix_range cannot disagree."""
    out = []
    ix0, ix1 = fig._get_epoch_ix_range()
    start, stop = fig._get_start_stop()
    if start != case.boundary_samples[ix0]:
        out.append(
            f"I1 start {start} != boundary_samples[{ix0}]"
            f"={case.boundary_samples[ix0]}"
        )
    if stop != case.boundary_samples[ix1]:
        out.append(
            f"I1 stop {stop} != boundary_samples[{ix1}]={case.boundary_samples[ix1]}"
        )
    want = int(case.lengths[ix0:ix1].sum())
    if stop - start != want:
        out.append(
            f"I1 window holds {stop - start} samples, epochs {ix0}:{ix1} hold {want}"
        )
    return out


def i2_nothing_invented(fig, case):
    """The loaded window is the source samples, in order, with nothing added."""
    out = []
    ix0, ix1 = fig._get_epoch_ix_range()
    start, stop = fig._get_start_stop()
    data, times = fig._load_data(start, stop)
    want = np.concatenate(case.source[ix0:ix1], axis=-1)
    if data.shape != want.shape:
        out.append(f"I2 loaded shape {data.shape} != source shape {want.shape}")
    elif not np.array_equal(data, want):
        bad = int(np.argmax(~np.isclose(data, want)))
        out.append(f"I2 loaded window differs from source, first at flat index {bad}")
    if np.isnan(np.asarray(data)).any():
        out.append("I2 NaN in the loaded window (padding leaked in)")
    if len(times) != stop - start:
        out.append(f"I2 times has {len(times)} entries, window is {stop - start}")
    if getattr(fig.mne, "data", None) is not None:
        if np.isnan(np.asarray(fig.mne.data)).any():
            out.append("I2 NaN in fig.mne.data")
    return out


def i3_whole_epochs(fig, case):
    """The view starts and ends exactly on epoch boundaries."""
    out = []
    bt = case.boundary_times
    t0 = float(fig.mne.t_start)
    t1 = float(fig.mne.t_start + fig.mne.duration)
    if not np.any(np.isclose(bt, t0, atol=1e-7)):
        near = bt[np.argmin(abs(bt - t0))]
        out.append(f"I3 t_start {t0:.9f} is not a boundary; nearest {near:.9f}")
    if not np.any(np.isclose(bt, t1, atol=1e-7)):
        near = bt[np.argmin(abs(bt - t1))]
        out.append(f"I3 t_end {t1:.9f} is not a boundary; nearest {near:.9f}")
    return out


def i4_visible_count(fig, case):
    """Exactly min(n_epochs, len) whole epochs are shown, never zero."""
    out = []
    ix0, ix1 = fig._get_epoch_ix_range()
    want = min(int(fig.mne.n_epochs), case.n_epochs)
    if ix1 - ix0 != want:
        out.append(
            f"I4 showing {ix1 - ix0} epochs, "
            f"n_epochs={fig.mne.n_epochs} of {case.n_epochs}"
        )
    if ix1 - ix0 < 1:
        out.append("I4 zero epochs visible")
    return out


def i5_in_range(fig, case):
    """Indices and times stay inside the data."""
    out = []
    ix0, ix1 = fig._get_epoch_ix_range()
    if not (0 <= ix0 < ix1 <= case.n_epochs):
        out.append(f"I5 epoch range {ix0}:{ix1} outside 0:{case.n_epochs}")
    if fig.mne.t_start < -TOL:
        out.append(f"I5 t_start {fig.mne.t_start} < 0")
    end = fig.mne.t_start + fig.mne.duration
    if end > case.boundary_times[-1] + 1.0 / case.sfreq:
        out.append(f"I5 view ends at {end}, data ends at {case.boundary_times[-1]}")
    if fig.mne.duration <= 0:
        out.append(f"I5 duration {fig.mne.duration} is not positive")
    return out


def _latency_of(case, x):
    """Latency relative to its own event, computed independently."""
    idx = int(
        np.clip(
            np.searchsorted(case.boundary_times[1:], x, side="right"),
            0,
            case.n_epochs - 1,
        )
    )
    offset = round((x - case.boundary_times[idx]) * case.sfreq)
    return idx, case.tmins[idx] + offset / case.sfreq


def i8_vline_latency(fig, case, backend):
    """Every drawn vline sits at one shared latency, inside a visible epoch."""
    out = []
    ix0, ix1 = fig._get_epoch_ix_range()
    vline = getattr(fig.mne, "vline", None)
    # both backends keep the artists around while hidden, so a stale position
    # only matters once the line is actually on screen
    if vline is None or not getattr(fig.mne, "vline_visible", False):
        return out
    if backend == "matplotlib":
        segs = np.asarray(vline.get_segments())
        if not len(segs):
            return out
        xs = segs[:, 0, 0]
    else:
        xs = np.array([vl.value() for vl in vline if vl.isVisible()], float)
        if not len(xs):
            return out

    latencies = []
    for x in xs:
        idx, lat = _latency_of(case, x)
        if not (ix0 <= idx < ix1):
            out.append(f"I8 vline at {x:.6f} lands in epoch {idx}, view is {ix0}:{ix1}")
        latencies.append(lat)
    if len(set(np.round(latencies, 6))) > 1:
        out.append(f"I8 vlines sit at different latencies: {_fmt(latencies)}")
    if latencies:
        lat = latencies[0]
        tol = 0.5 / case.sfreq
        tmaxs = case.tmins + (case.lengths - 1) / case.sfreq
        want = sum(
            1
            for i in range(ix0, ix1)
            if case.tmins[i] - tol <= lat <= tmaxs[i] + tol
        )
        if len(xs) != want:
            out.append(
                f"I8 {len(xs)} vlines drawn at latency {lat:.6f}, "
                f"{want} of the {ix1 - ix0} visible epochs reach it"
            )
    return out


def i9_qt_bookkeeping(fig, case):
    """Qt's own epoch index and line list stay in step with the view."""
    out = []
    ix0, ix1 = fig._get_epoch_ix_range()
    epoch_idx = np.atleast_1d(np.asarray(fig.mne.epoch_idx, int))
    if not np.array_equal(epoch_idx, np.arange(ix0, ix1)):
        out.append(f"I9 epoch_idx {_fmt(epoch_idx)} != arange({ix0}, {ix1})")
    vline = getattr(fig.mne, "vline", None)
    if vline is not None and len(vline) != len(epoch_idx):
        out.append(f"I9 {len(vline)} vline objects for {len(epoch_idx)} visible epochs")
    if not np.isclose(fig.mne.xmax, case.boundary_times[-1], atol=TOL):
        out.append(f"I9 xmax {fig.mne.xmax} != {case.boundary_times[-1]}")
    return out


def i11_mpl_hscroll_patch(fig, case):
    """The matplotlib scrollbar handle shows the window that is really drawn.

    ``hsel_patch`` is the only on-screen report of where in the recording the
    view sits, so it has to agree with ``t_start``/``duration`` *and* land on
    boundaries computed from the source arrays.
    """
    out = []
    patch = getattr(fig.mne, "hsel_patch", None)
    if patch is None:
        return out
    x0, _ = patch.get_xy()
    width = patch.get_width()
    if not np.isclose(x0, float(fig.mne.t_start), atol=1e-9):
        out.append(f"I11 hscroll patch starts at {x0}, view at {fig.mne.t_start}")
    if not np.isclose(width, float(fig.mne.duration), atol=1e-9):
        out.append(f"I11 hscroll patch is {width} wide, view is {fig.mne.duration}")
    bt = case.boundary_times
    if not np.any(np.isclose(bt, x0, atol=1e-7)):
        out.append(f"I11 hscroll patch starts at {x0:.9f}, not on a boundary")
    if not np.any(np.isclose(bt, x0 + width, atol=1e-7)):
        out.append(f"I11 hscroll patch ends at {x0 + width:.9f}, not on a boundary")
    if x0 + width > bt[-1] + 1e-7:
        out.append(f"I11 hscroll patch ends at {x0 + width:.9f} past data {bt[-1]:.9f}")
    return out


def i12_epoch_num_lookup(fig, case):
    """``_get_epoch_num_from_time`` answers for every x the view can show.

    Clicking a trace goes straight through this, so an unclamped searchsorted
    turns the last sample of the recording into an ``IndexError``.
    """
    out = []
    sel = np.asarray(fig.mne.inst.selection)
    ix0, ix1 = fig._get_epoch_ix_range()
    bt = case.boundary_times
    # Both backends treat an epoch as spanning ``(bt[k], bt[k + 1]]`` -- the
    # boundary itself belongs to the epoch on its left. That convention
    # predates this PR, so probe strictly inside each epoch, plus the two
    # extremes of the recording, which are the values that have to be clamped.
    probes = []
    for ix in range(ix0, ix1):
        span = bt[ix + 1] - bt[ix]
        probes += [(bt[ix] + 0.25 * span, ix), (bt[ix] + 0.75 * span, ix)]
    probes.append((bt[0], 0))
    probes.append((bt[-1], case.n_epochs - 1))
    for t, ix in probes:
        try:
            got = fig._get_epoch_num_from_time(t)
        except Exception as exc:
            out.append(
                f"I12 _get_epoch_num_from_time({t:.9f}) raised "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        want = sel[ix]
        if got != want:
            out.append(
                f"I12 _get_epoch_num_from_time({t:.9f}) = {got}, "
                f"that time is in epoch {ix} (number {want})"
            )
    return out


def i10_bad_epochs(fig, case):
    """Marked epochs are real members of the selection."""
    out = []
    sel = list(fig.mne.inst.selection)
    for num in fig.mne.bad_epochs:
        if num not in sel:
            out.append(f"I10 bad epoch {num} is not in selection {sel[:8]}...")
    if len(set(fig.mne.bad_epochs)) != len(fig.mne.bad_epochs):
        out.append(f"I10 duplicate entries in bad_epochs: {fig.mne.bad_epochs}")
    return out


_ALWAYS = (
    i0_model,
    i1_window_integrity,
    i2_nothing_invented,
    i3_whole_epochs,
    i4_visible_count,
    i5_in_range,
    i10_bad_epochs,
    i12_epoch_num_lookup,
)


def check_all(fig, case, backend):
    """Run every applicable invariant; return a flat list of violations."""
    out = []
    for fn in _ALWAYS:
        try:
            out.extend(fn(fig, case))
        except Exception:
            out.append(f"{fn.__name__} raised:\n{traceback.format_exc(limit=4)}")
    try:
        out.extend(i8_vline_latency(fig, case, backend))
    except Exception:
        out.append(f"i8_vline_latency raised:\n{traceback.format_exc(limit=4)}")
    if backend != "matplotlib":
        try:
            out.extend(i9_qt_bookkeeping(fig, case))
        except Exception:
            out.append(f"i9_qt_bookkeeping raised:\n{traceback.format_exc(limit=4)}")
    else:
        try:
            out.extend(i11_mpl_hscroll_patch(fig, case))
        except Exception:
            out.append(f"i11_mpl_hscroll_patch raised:\n{traceback.format_exc(limit=4)}")
    return out
