"""Shared helpers for slice A8 (visual verification).

Everything renders offscreen; every shot is accompanied by the numbers behind
it printed to stdout so the two can be compared.
"""

import os
import sys

import numpy as np

SHOTS = "D:/meta-mne-python-sprint/validation/browser_fuzz/shots"

sys.path.insert(0, "D:/mne-python")
sys.path.insert(0, "D:/meta-mne-python-sprint")

import mne  # noqa: E402
from validation.browser_fuzz.build import Spec, build  # noqa: E402

assert hasattr(mne.EpochsArray, "variable_duration"), "wrong mne on the path"


def app():
    from qtpy.QtWidgets import QApplication

    a = QApplication.instance()
    if a is None:
        a = QApplication([])
    return a


def rep_spec(lengths, n_channels=20, **kw):
    """A *representative* spec: 20 EEG channels of noise, not 3 clean ones."""
    return Spec(lengths=lengths, n_channels=n_channels, **kw)


def open_qt(case, **kwargs):
    mne.viz.set_browser_backend("qt")
    kw = dict(n_epochs=2, show=False, scalings=dict(eeg=3e-6))
    kw.update(kwargs)
    fig = case.epochs.plot(**kw)
    fig.test_mode = True
    return fig


def open_mpl(case, **kwargs):
    mne.viz.set_browser_backend("matplotlib")
    kw = dict(n_epochs=2, show=False, scalings=dict(eeg=3e-6))
    kw.update(kwargs)
    fig = case.epochs.plot(**kw)
    return fig


def shot_qt(fig, name, w=900, h=450):
    a = app()
    fig.resize(w, h)
    fig.show()
    a.processEvents()
    a.processEvents()
    path = os.path.join(SHOTS, name)
    fig.grab().save(path)
    print(f"  [saved] {path}")
    return path


def shot_mpl(fig, name, w=9.0, h=4.5, dpi=100):
    fig.set_size_inches(w, h)
    fig.canvas.draw()
    path = os.path.join(SHOTS, name)
    fig.savefig(path, dpi=dpi)
    print(f"  [saved] {path}")
    return path


def setx(fig, t0, t1):
    fig.mne.plt.setXRange(float(t0), float(t1), padding=0)
    app().processEvents()


def qt_numbers(fig, case, tag=""):
    """Print the numbers behind a Qt shot."""
    mne_ = fig.mne
    vb = mne_.viewbox
    (x0, x1) = (float(v) for v in vb.viewRange()[0])
    ix = fig._get_epoch_ix_range()
    print(f"  numbers{(' ' + tag) if tag else ''}:")
    print(f"    boundary_times = {np.round(case.boundary_times, 4).tolist()}")
    print(f"    viewRange x    = [{x0:.4f}, {x1:.4f}]  (t_start={mne_.t_start:.4f}, "
          f"duration={mne_.duration:.4f})")
    print(f"    epoch_ix_range = {ix}  epoch_idx={np.asarray(mne_.epoch_idx).tolist()}")
    ax = mne_.plt.getAxis("bottom")
    tv = getattr(ax, "_tickValues", None)
    print(f"    dark={mne_.dark}  butterfly={mne_.butterfly}  n_epochs={mne_.n_epochs}")
    return dict(view=(x0, x1), ix=ix)


def tick_report(fig):
    """What the bottom axis actually put on screen, in x order."""
    ax = fig.mne.plt.getAxis("bottom")
    vb = fig.mne.viewbox
    (x0, x1) = (float(v) for v in vb.viewRange()[0])
    w = vb.width() or 1
    # ask the axis for its own tick strings the same way it draws them
    spacings = ax.tickSpacing(x0, x1, w)
    out = []
    for spacing, offset in spacings:
        vals = ax.tickValues(x0, x1, w)
        break
    for scale_level in ax.tickValues(x0, x1, w):
        step, values = scale_level
        strings = ax.tickStrings(list(values), 1.0, step)
        for v, s in zip(values, strings):
            out.append((float(v), str(s), float(step)))
    out.sort()
    return out, (x0, x1), float(w)


def print_ticks(fig, label=""):
    ticks, (x0, x1), w = tick_report(fig)
    px = lambda t: (t - x0) / (x1 - x0) * w
    print(f"    ticks{(' ' + label) if label else ''} (view {x0:.3f}..{x1:.3f}, "
          f"{w:.0f}px):")
    prev = None
    for v, s, step in ticks:
        p = px(v)
        gap = "" if prev is None else f"  gap={p - prev:6.1f}px"
        flag = ""
        if prev is not None and (p - prev) < 30:
            flag = "   <-- CROWDED"
        print(f"      t={v:8.4f}  '{s}'  x={p:7.1f}px{gap}{flag}")
        prev = p
    return ticks
