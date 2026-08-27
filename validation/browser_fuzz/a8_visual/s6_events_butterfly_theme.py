"""A8-6: event lines with per-epoch tmin, butterfly mode, a resize, both themes.

Every panel prints the numbers behind it. The theme section establishes what
was actually rendered (``mne.dark``) before drawing any conclusion, because
``theme="dark"`` alone does not repaint the palette under an offscreen platform.
"""

import sys

import numpy as np
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QApplication

from vis import app, build, open_qt, rep_spec, shot_qt, Spec
from validation.browser_fuzz.build import build as _b


def event_report(fig, case, tag):
    """Compare the drawn event lines against the latency-0 position per epoch."""
    mne_ = fig.mne
    print(f"  events {tag}:")
    if mne_.event_times is None:
        print("    (none)")
        return
    # where t = 0 falls inside each epoch, from the source arrays only
    want = []
    for ix in range(case.n_epochs):
        tmin = float(case.tmins[ix])
        dur = case.lengths[ix] / case.sfreq
        if -tmin < 0 or -tmin > dur - 1 / case.sfreq + 1e-9:
            want.append(None)  # t=0 is outside this epoch
        else:
            want.append(float(case.boundary_times[ix]) - tmin)
    print(f"    tmins            = {np.round(case.tmins, 3).tolist()}")
    print(f"    expected t=0 at  = {[None if v is None else round(v, 4) for v in want]}")
    print(f"    mne.event_times  = {np.round(mne_.event_times, 4).tolist()}")
    drawn = sorted(
        float(ln.value())
        for ln in fig.mne.viewbox.addedItems
        if type(ln).__name__ == "EventLine"
    )
    print(f"    EventLine x      = {np.round(drawn, 4).tolist()}")
    exp = sorted(v for v in want if v is not None)
    ok = len(exp) == len(mne_.event_times) and np.allclose(
        exp, sorted(mne_.event_times), atol=1e-9
    )
    print(f"    event_times match the per-epoch t=0 positions: {ok}")


# --- events with per-epoch tmin -------------------------------------------
for label, tmin, lengths, fixed in [
    ("ragged_tmin_mixed", "mixed", (100, 250, 75, 180), False),
    ("ragged_tmin_neg", "negative", (100, 250, 75, 180), False),
    ("fixed_tmin_zero", "zero", (150, 150, 150, 150), True),
    ("equal_tmin_mixed", "mixed", (150, 150, 150, 150), False),
]:
    print(f"\n=== events {label} lengths={lengths} tmin={tmin} ===")
    case = build(rep_spec(lengths, n_channels=20, tmin=tmin, force_fixed=fixed))
    fig = open_qt(case, n_epochs=4, events=True)
    shot_qt(fig, f"a8_events_{label}.png")
    print(f"    boundary_times = {np.round(case.boundary_times, 4).tolist()}")
    event_report(fig, case, label)
    fig.close()

# --- butterfly + resize ----------------------------------------------------
for label, lengths, fixed in [
    ("ragged", (100, 250, 75, 180), False),
    ("fixed", (150, 150, 150, 150), True),
]:
    print(f"\n=== butterfly/resize {label} ===")
    case = build(rep_spec(lengths, n_channels=20, mixed_types=True, force_fixed=fixed))
    fig = open_qt(case, n_epochs=4, butterfly=True)
    shot_qt(fig, f"a8_butterfly_{label}_900x450.png", w=900, h=450)
    print(f"    butterfly={fig.mne.butterfly}  view="
          f"{[round(float(v), 4) for v in fig.mne.viewbox.viewRange()[0]]}")
    # resize emits no range signal: anything pixel-positioned can go stale
    shot_qt(fig, f"a8_butterfly_{label}_400x300.png", w=400, h=300)
    shot_qt(fig, f"a8_butterfly_{label}_1500x700.png", w=1500, h=700)
    ax = fig.mne.plt.getAxis("bottom")
    print(f"    after resizes: labels={ax.get_labels()}  view="
          f"{[round(float(v), 4) for v in fig.mne.viewbox.viewRange()[0]]}")
    fig.close()

    # non-butterfly resize sweep for the boundary/label geometry
    fig = open_qt(case, n_epochs=4)
    for w, h in ((900, 450), (300, 200), (1500, 700), (900, 450)):
        shot_qt(fig, f"a8_resize_{label}_{w}x{h}.png", w=w, h=h)
        vb = fig.mne.viewbox
        x0, x1 = (float(v) for v in vb.viewRange()[0])
        bw = vb.width() or 1
        sep = sorted(
            float(i.value())
            for i in vb.addedItems
            if type(i).__name__ == "InfiniteLine" and i.angle == 90
        )
        px = [round((t - x0) / (x1 - x0) * bw, 2) for t in sep]
        want = [
            round((float(t) - x0) / (x1 - x0) * bw, 2)
            for t in case.boundary_times[1:-1]
        ]
        print(f"    {w}x{h}: vb {bw:.0f}px  separators px={px}  expected={want}  "
              f"match={px == want}")
    fig.close()

# --- themes ----------------------------------------------------------------
print("\n=== themes ===")
case = build(rep_spec((100, 250, 75, 180), n_channels=20))
for theme in ("light", "dark"):
    fig = open_qt(case, n_epochs=4, theme=theme)
    bg = fig.palette().color(fig.backgroundRole()).getRgbF()[:3]
    print(f"  theme={theme!r}: mne.dark={fig.mne.dark}  widget bg rgb="
          f"{tuple(round(v, 3) for v in bg)}")
    shot_qt(fig, f"a8_theme_{theme}_asrequested.png")
    fig.close()

# force a genuinely dark palette so mne.dark is really True, then re-render
a = app()
a.setStyle("Fusion")
pal = QPalette()
for role in (QPalette.Window, QPalette.Base, QPalette.Button, QPalette.AlternateBase):
    pal.setColor(role, QColor(30, 30, 30))
for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
    pal.setColor(role, QColor(230, 230, 230))
a.setPalette(pal)
fig = open_qt(case, n_epochs=4, theme="dark")
bg = fig.palette().color(fig.backgroundRole()).getRgbF()[:3]
print(f"  forced dark palette: mne.dark={fig.mne.dark}  widget bg rgb="
      f"{tuple(round(v, 3) for v in bg)}")
shot_qt(fig, "a8_theme_dark_forced.png")
ax = fig.mne.plt.getAxis("bottom")
print(f"    labels={ax.get_labels()}")
fig.close()
