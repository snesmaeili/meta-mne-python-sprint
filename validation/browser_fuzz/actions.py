"""The action alphabet the sweep drives the browser with.

Every action is a named callable so a failing sequence prints as a script the
reader can retype. Nothing here closes the figure or opens a modal dialog, so a
long random walk stays inside one browser.
"""

import numpy as np

# Keys both backends bind to the same meaning. Arrow keys move the view, home
# and end change how many epochs it holds.
NAV_KEYS = (
    "left",
    "right",
    "shift+left",
    "shift+right",
    "home",
    "end",
)

# Keys that change what is drawn rather than which samples are shown.
DISPLAY_KEYS = (
    "up",
    "down",
    "pageup",
    "pagedown",
    "b",  # butterfly
    "d",  # DC removal
    "s",  # scalebars
    "0",  # zero line
    "t",  # time format
    "+",
    "-",
)

# Keys that open a child window. Cheap under Agg / offscreen, but kept apart so
# a fuzz seed can be reproduced without them.
WINDOW_KEYS = ("?", "j", "h")


def _click_epoch(fig, case, which):
    """Click in the middle of a visible epoch (marks it bad in matplotlib)."""

    def action():
        ix0, ix1 = fig._get_epoch_ix_range()
        ix = int(np.clip(ix0 + which, ix0, ix1 - 1))
        x = (case.boundary_times[ix] + case.boundary_times[ix + 1]) / 2
        y = fig.mne.traces[0].get_ydata()[0]
        fig._fake_click((x, y), xform="data")

    return action


def _click_latency(fig, case, frac):
    """Click at a fraction into the first visible epoch (places a vline)."""

    def action():
        ix0, _ = fig._get_epoch_ix_range()
        span = case.boundary_times[ix0 + 1] - case.boundary_times[ix0]
        x = case.boundary_times[ix0] + frac * span
        fig._fake_click((x, 0.5), xform="data")

    return action


def _key(fig, key):
    def action():
        fig._fake_keypress(key)

    return action


# -- matplotlib scrollbar gestures ------------------------------------------
#
# The Qt browser gets its scrollbar exercised through ``hscroll``; the
# matplotlib one only moves when a real click or drag lands in ``ax_hscroll``,
# which is a different code path (``_check_update_hscroll_clicked`` and
# ``_mouse_move``) from the arrow keys.


def _mpl_hscroll_click(fig, case, frac):
    """Left-click in the horizontal scrollbar at ``frac`` of the whole span."""

    def action():
        x = float(frac) * case.boundary_times[-1]
        ax = fig.mne.ax_hscroll
        fig._fake_click((x, 0.5), ax=ax, xform="data", kind="press")
        fig._fake_click((x, 0.5), ax=ax, xform="data", kind="release")

    return action


def _mpl_hscroll_drag(fig, case, frac_from, frac_to):
    """Press in the scrollbar at one fraction and drag to another."""

    def action():
        span = case.boundary_times[-1]
        x0, x1 = float(frac_from) * span, float(frac_to) * span
        ax = fig.mne.ax_hscroll
        fig._fake_click((x0, 0.5), ax=ax, xform="data", kind="press")
        fig._fake_click((x1, 0.5), ax=ax, xform="data", kind="motion")
        fig._fake_click((x1, 0.5), ax=ax, xform="data", kind="release")

    return action


def _mpl_click_edge(fig, case, where):
    """Click on the very first or very last sample of the visible window.

    ``_get_epoch_num_from_time`` searchsorts ``boundary_times[1:]`` without
    clamping, so the last drawable x is the interesting one (seed suspect S6).
    """

    def action():
        ix0, ix1 = fig._get_epoch_ix_range()
        if where == "last":
            x = case.boundary_times[ix1] - 1.0 / case.sfreq
        elif where == "past_end":
            x = case.boundary_times[ix1]
        else:
            x = case.boundary_times[ix0]
        y = fig.mne.traces[0].get_ydata()[0]
        fig._fake_click((x, y), xform="data")

    return action


def _qt_hscroll(fig, step):
    def action():
        fig.hscroll(step)

    return action


def _qt_duration(fig, step):
    def action():
        fig.change_duration(step=step)

    return action


def _qt_setxrange(fig, case, lo_frac, hi_frac):
    """Set a range that is deliberately not aligned to epoch boundaries."""

    def action():
        span = case.boundary_times[-1]
        lo, hi = lo_frac * span, hi_frac * span
        if hi - lo < 1.0 / case.sfreq:
            hi = lo + 1.0 / case.sfreq
        fig.mne.plt.setXRange(lo, hi, padding=0)

    return action


def _qt_vline(fig, case, frac):
    def action():
        ix0, _ = fig._get_epoch_ix_range()
        span = case.boundary_times[ix0 + 1] - case.boundary_times[ix0]
        fig._add_vline(case.boundary_times[ix0] + frac * span)

    return action


def build_alphabet(fig, case, backend, include_windows=False, include_setxrange=False):
    """Return ``[(name, callable), ...]`` valid for this figure.

    ``setxrange`` is opt-in: a raw range that is not epoch-aligned breaks an
    assumption the Qt browser states about itself, on the fixed path too, and
    it is not yet known whether a real gesture can reach one. See N2 in
    FINDINGS.md.
    """
    acts = []
    # Qt binds a different, smaller key set than matplotlib and raises KeyError
    # on anything outside it, so ask the figure what it actually accepts rather
    # than trusting a hardcoded list. Matplotlib has no such table and its
    # handler ignores keys it does not know.
    bound = getattr(fig.mne, "keyboard_shortcuts", None)

    def _is_bound(key):
        if bound is None:
            return True
        return key.split("+")[-1] in bound

    for key in NAV_KEYS + DISPLAY_KEYS:
        if _is_bound(key):
            acts.append((f"key:{key}", _key(fig, key)))
    if include_windows:
        for key in WINDOW_KEYS:
            if _is_bound(key):
                acts.append((f"key:{key}", _key(fig, key)))

    if backend == "matplotlib":
        for which in (0, 1):
            acts.append((f"click_epoch:{which}", _click_epoch(fig, case, which)))
        for frac in (0.1, 0.5, 0.9):
            acts.append((f"click_latency:{frac}", _click_latency(fig, case, frac)))
        for frac in (0.0, 0.13, 0.5, 0.87, 1.0):
            acts.append((f"hscroll_click:{frac}", _mpl_hscroll_click(fig, case, frac)))
        for lo, hi in ((0.5, 0.0), (0.0, 1.0), (0.5, 1.0), (1.0, 0.5), (0.2, 0.8)):
            acts.append(
                (f"hscroll_drag:{lo}->{hi}", _mpl_hscroll_drag(fig, case, lo, hi))
            )
        for where in ("first", "last", "past_end"):
            acts.append((f"click_edge:{where}", _mpl_click_edge(fig, case, where)))
    else:
        for step in ("left", "right", "-full", "+full"):
            acts.append((f"hscroll:{step}", _qt_hscroll(fig, step)))
        for step in (1, -1):
            acts.append((f"change_duration:{step:+d}", _qt_duration(fig, step)))
        if include_setxrange:
            for lo, hi in ((0.0, 0.37), (0.21, 0.64), (0.55, 1.0)):
                acts.append((f"setxrange:{lo}-{hi}", _qt_setxrange(fig, case, lo, hi)))
        for frac in (0.1, 0.5, 0.9):
            acts.append((f"vline:{frac}", _qt_vline(fig, case, frac)))
    return acts
