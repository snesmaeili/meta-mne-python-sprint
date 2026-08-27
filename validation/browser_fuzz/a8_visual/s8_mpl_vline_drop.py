"""A8-8: matplotlib vlines through the same gesture as F1, and dropped-epoch
tick labels on both backends.
"""

import numpy as np
import matplotlib.pyplot as plt

from vis import app, build, open_mpl, open_qt, rep_spec, shot_mpl, shot_qt


def mpl_vline_report(fig, case, tag):
    mne_ = fig.mne
    v = mne_.vline
    print(f"  {tag}: xlim="
          f"{[round(float(x), 4) for x in fig.mne.ax_main.get_xlim()]}")
    if v is None:
        print("    no vline")
        return
    segs = v.get_segments()
    xs = [float(np.asarray(s)[0, 0]) for s in segs]
    print(f"    {len(xs)} vline segment(s) drawn; visible={v.get_visible()}; "
          f"label={fig.mne.vline_text.get_text()!r}")
    for x in sorted(xs):
        ix = int(np.clip(np.searchsorted(case.boundary_times[1:], x, side="right"),
                         0, case.n_epochs - 1))
        lat = case.tmins[ix] + round((x - case.boundary_times[ix]) * case.sfreq) / case.sfreq
        print(f"    x={x:.4f} -> epoch {ix} latency {lat:.4f}")


print("=== matplotlib vline through a scroll ===")
for label, lengths, fixed in [
    ("ragged", (100, 250, 75, 180), False),
    ("fixed", (150, 150, 150, 150), True),
]:
    case = build(rep_spec(lengths, n_channels=20, force_fixed=fixed))
    fig = open_mpl(case, n_epochs=2)
    print(f"\n {label}: boundary_times={np.round(case.boundary_times, 4).tolist()}")
    try:
        fig._show_vline(0.9)
        fig.canvas.draw()
        shot_mpl(fig, f"a8_mplvline_{label}_1_before.png")
        mpl_vline_report(fig, case, "after _show_vline(0.9)")
        fig._fake_keypress("right")
        fig.canvas.draw()
        shot_mpl(fig, f"a8_mplvline_{label}_2_after_scroll.png")
        mpl_vline_report(fig, case, "after key 'right'")
    except Exception as exc:
        print(f"    EXCEPTION {type(exc).__name__}: {exc}")
    plt.close(fig)

print("\n=== dropped epochs: do the tick labels name the surviving epochs? ===")
from validation.browser_fuzz.build import Spec

for label, spec in [
    ("drop_noncontiguous_ragged",
     Spec(lengths=(100, 250, 75, 180, 120), n_channels=20, drop=(1, 3))),
    ("drop_noncontiguous_fixed",
     Spec(lengths=(150,) * 5, n_channels=20, drop=(1, 3), force_fixed=True)),
]:
    case = build(spec)
    print(f"\n {label}: kept selection={case.epochs.selection.tolist()} "
          f"boundaries={np.round(case.boundary_times, 4).tolist()}")
    figq = open_qt(case, n_epochs=case.n_epochs)
    shot_qt(figq, f"a8_drop_{label}_qt.png")
    ax = figq.mne.plt.getAxis("bottom")
    print(f"    qt  labels={ax.get_labels()}  (expect "
          f"{[str(s) for s in case.epochs.selection]})")
    figq.close()
    figm = open_mpl(case, n_epochs=case.n_epochs)
    shot_mpl(figm, f"a8_drop_{label}_mpl.png")
    ticks = [
        (round(float(t.get_loc()), 4), t.label1.get_text())
        for t in figm.mne.ax_main.xaxis.get_minor_ticks()
        if t.label1.get_text()
    ]
    print(f"    mpl labels={ticks}")
    print(f"    mpl midpoints={np.round(figm.mne.midpoints, 4).tolist()}")
    plt.close(figm)
