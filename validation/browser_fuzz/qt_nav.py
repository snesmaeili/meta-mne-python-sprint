"""Qt navigation / vline actions and invariants (slice A4).

Kept out of the shared ``actions.py`` and ``invariants.py`` so the slices do
not collide. Everything here drives the main plot's horizontal navigation and
the epoch vlines, plus the two things a range signal never reports: a window
resize, and the crosshair readout.

Use with :func:`runner.run` by merging the alphabet::

    from validation.browser_fuzz import qt_nav
    table = dict(actions.build_alphabet(fig, case, "qt"))
    table.update(qt_nav.build_nav_alphabet(fig, case))
"""

import numpy as np
from qtpy.QtCore import QPointF
from qtpy.QtWidgets import QApplication


# -- extra invariants -------------------------------------------------------
def n1_view_cache_agrees(fig, case):
    """``t_start``/``duration`` are caches of the authoritative view range.

    ``mne.viewbox.viewRange()`` is what pyqtgraph draws; ``mne.t_start`` and
    ``mne.duration`` are refreshed only inside ``_xrange_changed``, which runs
    only when pyqtgraph decides the range actually changed.
    """
    out = []
    vb = getattr(fig.mne, "viewbox", None)
    if vb is None:
        return out
    (t0, t1) = (float(v) for v in vb.viewRange()[0])
    c0 = float(fig.mne.t_start)
    c1 = c0 + float(fig.mne.duration)
    if abs(t0 - c0) > 1e-9 or abs(t1 - c1) > 1e-9:
        out.append(
            f"N1 viewRange [{t0:.9f}, {t1:.9f}] but the cache says "
            f"[{c0:.9f}, {c1:.9f}]"
        )
    return out


def n2_vline_label_matches_line(fig, case):
    """Each visible vline's label must read the latency of its own position.

    ``VLineLabel.valueChanged`` returns early while the label is hidden, so a
    line that is repositioned while hidden and shown again carries the old
    text. The label is what the user reads off the screen.
    """
    out = []
    vline = getattr(fig.mne, "vline", None)
    if vline is None or not fig.mne.is_epochs:
        return out
    tol = 0.5 / case.sfreq
    for i, vl in enumerate(vline):
        if not vl.isVisible():
            continue
        x = float(vl.value())
        idx = int(
            np.clip(
                np.searchsorted(case.boundary_times[1:], x, side="right"),
                0,
                case.n_epochs - 1,
            )
        )
        want = case.tmins[idx] + round((x - case.boundary_times[idx]) * case.sfreq) / (
            case.sfreq
        )
        try:
            shown = float(vl.label.textItem.toPlainText().split()[0])
        except (ValueError, IndexError):
            out.append(f"N2 vline {i} label is not a number")
            continue
        if abs(shown - want) > tol:
            out.append(
                f"N2 vline {i} sits at latency {want:.6f} but its label reads "
                f"{shown:.6f}"
            )
    return out


def n3_vline_count(fig, case):
    """One vline object per visible epoch, so no epoch is silently skipped."""
    out = []
    vline = getattr(fig.mne, "vline", None)
    if vline is None or not fig.mne.is_epochs:
        return out
    ix0, ix1 = fig._get_epoch_ix_range()
    if len(vline) != ix1 - ix0:
        out.append(
            f"N3 {len(vline)} vline objects for {ix1 - ix0} visible epochs "
            f"({ix0}:{ix1})"
        )
    return out


def check_nav(fig, case):
    """Run the navigation-specific invariants."""
    out = []
    for fn in (n1_view_cache_agrees, n2_vline_label_matches_line, n3_vline_count):
        try:
            out.extend(fn(fig, case))
        except Exception as exc:  # pragma: no cover - reported, not raised
            out.append(f"{fn.__name__} raised {type(exc).__name__}: {exc}")
    return out


# -- actions ----------------------------------------------------------------
def _vline_at(fig, case, latency):
    """Place a vline at an absolute latency in the first visible epoch.

    Unlike ``actions.vline:<frac>`` this does not scale with the epoch, so the
    same latency can exist in some epochs and not in others.
    """

    def action():
        ix0, _ = fig._get_epoch_ix_range()
        x = case.boundary_times[ix0] + (latency - case.tmins[ix0])
        fig._add_vline(float(x))

    return action


def _resize(fig, width, height):
    """Resize the window. No range signal is emitted, so pixel-positioned
    items only refresh if something listens for the resize itself."""

    def action():
        fig.resize(width, height)
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    return action


def _crosshair(fig, case, frac):
    """Move the crosshair to a fraction across the visible window."""

    def action():
        fig._toggle_crosshair(True)
        t0 = float(fig.mne.t_start)
        t1 = t0 + float(fig.mne.duration)
        x = t0 + frac * (t1 - t0)
        pos = fig.mne.viewbox.mapViewToScene(QPointF(x, fig.mne.traces[0].ypos))
        fig._mouse_moved(pos)

    return action


def build_nav_alphabet(fig, case):
    """Return ``[(name, callable), ...]`` for the navigation surface."""
    acts = []
    for latency in (0.0, 0.3, 0.74, 0.9, 1.5):
        acts.append((f"vline_at:{latency}", _vline_at(fig, case, latency)))
    for w, h in ((900, 450), (400, 300), (160, 120), (1600, 900)):
        acts.append((f"resize:{w}x{h}", _resize(fig, w, h)))
    for frac in (0.0, 0.5, 0.999, 1.0):
        acts.append((f"crosshair:{frac}", _crosshair(fig, case, frac)))
    return acts
