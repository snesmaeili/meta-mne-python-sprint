"""Shared helpers for the A2 (matplotlib display state) slice.

Everything here reads *drawn* artist state and compares it against arrays the
harness computed from the source data, never against the browser's own model.
"""

import numpy as np

import mne

from ..build import build  # noqa: F401
from ..runner import close_fig, open_browser  # noqa: F401


def assert_env():
    assert hasattr(mne.EpochsArray, "variable_duration"), (
        f"wrong mne on path: {mne.__file__}"
    )


def full_xy(line):
    """Underlying (unmasked) x and y of a Line2D drawn by ``_draw_traces``."""
    x = np.ma.getdata(line.get_xdata())
    y = np.ma.getdata(line.get_ydata())
    return np.asarray(x, float), np.asarray(y, float)


def drawn_samples(fig, case):
    """Global sample indices actually carried by the primary traces.

    Derived from the x values on the artists and ``case.sfreq`` -- the browser's
    ``times`` / ``decim`` attributes are never consulted.
    """
    line = fig.mne.traces[0]
    x, _ = full_xy(line)
    ft = float(fig.mne.first_time)
    return np.round((x - ft) * case.sfreq).astype(int)


def per_epoch_phase(samples, case):
    """For each epoch, the within-epoch index of its first drawn sample.

    Returns ``{epoch_index: (first_within_epoch_index, n_drawn)}`` for every
    epoch that has at least one drawn sample.
    """
    out = {}
    bs = case.boundary_samples
    for k in range(case.n_epochs):
        lo, hi = bs[k], bs[k + 1]
        inside = samples[(samples >= lo) & (samples < hi)]
        if len(inside):
            out[k] = (int(inside[0] - lo), int(len(inside)))
    return out


def trace_matches_source(fig, case, *, atol=1e-9):
    """Check every drawn y value equals ``offset - source * scale_factor``.

    Returns a list of complaint strings.
    """
    bad = []
    samples = drawn_samples(fig, case)
    scale = float(fig.mne.scale_factor)
    picks = np.asarray(fig.mne.picks)
    offsets = np.asarray(fig.mne.trace_offsets)
    if fig.mne.butterfly and fig.mne.ch_selections is None:
        offsets = offsets[picks]
    # map global sample -> (epoch, within index)
    bs = case.boundary_samples
    ep = np.searchsorted(bs[1:], samples, side="right")
    within = samples - bs[ep]
    # scalings per channel type, as the browser normalises by 2 * norm
    for ii, line in enumerate(fig.mne.traces):
        if ii >= len(picks):
            break
        ch = int(picks[ii])
        _, y = full_xy(line)
        if len(y) != len(samples):
            bad.append(f"trace {ii}: {len(y)} y values for {len(samples)} x values")
            continue
        src = np.array([case.source[e][ch, w] for e, w in zip(ep, within)])
        drawn = (offsets[ii] - y) / scale
        # the browser removes DC and divides by 2*norm, so the drawn trace must
        # be an *affine* image of the source: fit it and look at the residual
        if len(src) < 3 or np.ptp(src) == 0:
            continue
        A = np.column_stack([src, np.ones_like(src)])
        coef, *_ = np.linalg.lstsq(A, drawn, rcond=None)
        resid = drawn - A @ coef
        rel = np.abs(resid).max() / max(np.ptp(drawn), 1e-30)
        if rel > 1e-6:
            worst = int(np.argmax(np.abs(resid)))
            bad.append(
                f"trace {ii} (ch {ch}): drawn y is not an affine image of the "
                f"source -- max rel residual {rel:.3g} at drawn index {worst} "
                f"(global sample {samples[worst]}, epoch {ep[worst]}, "
                f"within {within[worst]})"
            )
    return bad
