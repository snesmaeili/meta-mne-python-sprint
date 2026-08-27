"""A8-5: what the A4-1 overview-bar click looks like on screen.

An overview-bar click picks a start epoch and then reuses the *current*
``mne.duration`` as the width. With ragged epochs the duration of n epochs
depends on which n, so the window can be wider than the epochs it loads: the
right-hand slice of the plot is empty while the tick labels still name the
epochs it claims to show.
"""

import numpy as np
from qtpy.QtCore import QPoint, Qt
from qtpy.QtTest import QTest

from vis import app, build, open_qt, rep_spec, shot_qt


def ov_click(fig, frac):
    ob = fig.mne.overview_bar
    vp = ob.viewport()
    QTest.mouseClick(
        vp, Qt.LeftButton, pos=QPoint(int(vp.width() * frac), vp.height() // 2)
    )
    app().processEvents()


def window_report(fig, case, tag):
    vb = fig.mne.viewbox
    x0, x1 = (float(v) for v in vb.viewRange()[0])
    ix0, ix1 = fig._get_epoch_ix_range()
    lo = float(case.boundary_times[ix0])
    hi = float(case.boundary_times[ix1]) - 1 / case.sfreq
    blank = max(0.0, (x1 - hi)) + max(0.0, (lo - x0))
    ax = fig.mne.plt.getAxis("bottom")
    print(f"  {tag}")
    print(f"    view      = [{x0:.4f}, {x1:.4f}]  width {x1 - x0:.4f}s")
    print(f"    epochs    = {ix0}:{ix1}  loaded data spans [{lo:.4f}, {hi:.4f}]")
    print(f"    axis says = {ax.get_labels()}")
    print(
        f"    blank     = {blank:.4f}s  = {100 * blank / (x1 - x0):.1f}% of the window"
    )
    aligned = any(abs(x0 - b) < 1e-9 for b in case.boundary_times) and any(
        abs(x1 - b) < 1e-9 for b in case.boundary_times
    )
    print(f"    both ends on a boundary: {aligned}")
    return blank / (x1 - x0)


for label, lengths, fixed in [
    ("ragged", (100, 250, 75, 180), False),
    ("fixed", (150, 150, 150, 150), True),
]:
    print(f"\n=== overview click {label} lengths={lengths} ===")
    case = build(rep_spec(lengths, n_channels=20, force_fixed=fixed))
    fig = open_qt(case, n_epochs=2)
    print(f"  boundary_times = {np.round(case.boundary_times, 4).tolist()}")
    shot_qt(fig, f"a8_ovclick_{label}_0_open.png")
    window_report(fig, case, "at open")

    ov_click(fig, 0.99)
    shot_qt(fig, f"a8_ovclick_{label}_1_click_far_right.png")
    window_report(fig, case, "after clicking the far right of the overview bar")

    ov_click(fig, 0.0)
    shot_qt(fig, f"a8_ovclick_{label}_2_click_far_left.png")
    window_report(fig, case, "then clicking the far left")

    # the n_epochs=1 case from A4-1: 1.49 s of a 2.5 s epoch off-screen
    fig.close()

    fig = open_qt(case, n_epochs=1)
    ov_click(fig, 0.25)
    shot_qt(fig, f"a8_ovclick_{label}_3_one_epoch_25pct.png")
    window_report(fig, case, "n_epochs=1, click 25% along the bar")
    fig.close()
