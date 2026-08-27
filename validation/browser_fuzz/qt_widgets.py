"""Qt widget / dialog / toolbar actions and invariants (slice A5).

Kept in its own module so the shared ``actions.py`` and ``invariants.py`` stay
free for the other slices. Everything here drives the parts of the browser that
are *not* the main plot: the toolbar, the horizontal scrollbar, the overview
bar, the settings dialog, and the channel axis.
"""

import numpy as np
from qtpy.QtCore import Qt, QPoint
from qtpy.QtTest import QTest

TOOLBAR_NAMES = (
    "Show fewer time points",
    "Show more time points",
    "Show fewer channels",
    "Show more channels",
    "Reduce amplitude",
    "Increase amplitude",
    "Show projectors",
    "Toggle crosshair",
    "Settings",
    "Help",
)


# -- extra invariants -------------------------------------------------------
def w1_view_matches_data(fig, case):
    """What the x range shows is what the traces actually hold.

    ``_get_epoch_ix_range`` loads ``n_epochs`` whole epochs starting at
    ``t_start``; the viewbox range is set separately. When a widget sets a
    range that is a fixed number of seconds rather than the span of those
    epochs, the two disagree and the browser draws blank space, or hides data
    it has loaded, with nothing to say so.
    """
    out = []
    traces = getattr(fig.mne, "traces", None)
    if not traces:
        return out
    xd = np.asarray(traces[0].get_xdata(), float)
    if not xd.size:
        return out
    t0 = float(fig.mne.t_start)
    t1 = t0 + float(fig.mne.duration)
    step = 1.0 / case.sfreq
    if xd.min() > t0 + 0.5 * step:
        out.append(
            f"W1 view starts at {t0:.6f} but the drawn data starts at "
            f"{xd.min():.6f}: {xd.min() - t0:.6f} s of blank on the left"
        )
    if xd.max() < t1 - 1.5 * step:
        out.append(
            f"W1 view ends at {t1:.6f} but the drawn data ends at "
            f"{xd.max():.6f}: {t1 - xd.max():.6f} s of blank on the right"
        )
    if xd.max() > t1 + 0.5 * step:
        out.append(
            f"W1 view ends at {t1:.6f} but the traces hold data out to "
            f"{xd.max():.6f}: {xd.max() - t1:.6f} s loaded and not shown"
        )
    return out


def w2_window_spans_its_epochs(fig, case):
    """The visible seconds equal the seconds the loaded epochs really hold."""
    out = []
    ix0, ix1 = fig._get_epoch_ix_range()
    want = float(case.boundary_times[ix1] - case.boundary_times[ix0])
    got = float(fig.mne.duration)
    if not np.isclose(got, want, atol=0.5 / case.sfreq):
        out.append(
            f"W2 duration {got:.6f} s but epochs {ix0}:{ix1} span {want:.6f} s"
        )
    return out


def check_widgets(fig, case):
    """Run the widget-specific invariants."""
    out = []
    for fn in (w1_view_matches_data, w2_window_spans_its_epochs):
        try:
            out.extend(fn(fig, case))
        except Exception as exc:  # pragma: no cover - reported, not raised
            out.append(f"{fn.__name__} raised {type(exc).__name__}: {exc}")
    return out


# -- actions ----------------------------------------------------------------
def _toolbar(fig, name):
    def action():
        fig._fake_click_on_toolbar_action(name, wait_after=0)

    return action


def _overview_click(fig, frac):
    def action():
        ob = fig.mne.overview_bar
        vp = ob.viewport()
        pos = QPoint(int(vp.width() * frac), vp.height() // 2)
        QTest.mouseClick(vp, Qt.LeftButton, pos=pos)

    return action


def _hscroll_bar(fig, frac):
    def action():
        bar = fig.mne.ax_hscroll
        bar.setValue(int(round(frac * bar.maximum())))

    return action


def _hscroll_step(fig, add):
    def action():
        bar = fig.mne.ax_hscroll
        act = (
            bar.SliderAction.SliderSingleStepAdd
            if add
            else bar.SliderAction.SliderSingleStepSub
        )
        bar.triggerAction(act)

    return action


def _settings(fig, what, value=None):
    def action():
        dlg = fig.mne.fig_settings
        if dlg is None:
            return
        if what == "downsampling":
            dlg.downsampling_box.setValue(value)
        elif what == "ds_method":
            dlg.ds_method_cmbx.setCurrentText(value)
        elif what == "scroll_sensitivity":
            dlg.scroll_sensitivity_slider.setValue(value)
        elif what == "units":
            dlg.physical_units_cmbx.setCurrentText(value)
        elif what == "sensitivity":
            for ct, box in dlg.ch_sensitivity_spinboxes.items():
                box.setValue(value)
                break
        elif what == "scaling":
            for ct, box in dlg.ch_scaling_spinboxes.items():
                box.setValue(max(value, 1e-9))
                break
        elif what == "monitor_reset":
            dlg._reset_monitor_spinboxes()
        else:
            raise ValueError(what)

    return action


def _overview_mode(fig, mode):
    def action():
        fig._overview_mode_changed(mode)

    return action


def build_widget_alphabet(fig, case):
    """Return ``[(name, callable), ...]`` for the widget surface."""
    acts = []
    for name in TOOLBAR_NAMES:
        acts.append((f"toolbar:{name}", _toolbar(fig, name)))
    for frac in (0.0, 0.15, 0.35, 0.5, 0.62, 0.8, 0.95, 1.0):
        acts.append((f"ov_click:{frac}", _overview_click(fig, frac)))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        acts.append((f"hscroll_bar:{frac}", _hscroll_bar(fig, frac)))
    acts.append(("hscroll_bar:+step", _hscroll_step(fig, True)))
    acts.append(("hscroll_bar:-step", _hscroll_step(fig, False)))
    for mode in ("channels", "zscore", "empty"):
        acts.append((f"overview_mode:{mode}", _overview_mode(fig, mode)))
    for what, value in (
        ("downsampling", 3),
        ("downsampling", 0),
        ("ds_method", "mean"),
        ("ds_method", "peak"),
        ("scroll_sensitivity", 400),
        ("units", "/ cm"),
        ("units", "/ mm"),
        ("sensitivity", 12.5),
        ("scaling", 2.0),
        ("monitor_reset", None),
    ):
        acts.append((f"settings:{what}={value}", _settings(fig, what, value)))
    return acts
