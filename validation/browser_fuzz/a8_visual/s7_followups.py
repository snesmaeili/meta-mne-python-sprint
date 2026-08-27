"""A8-7: follow-ups.

1. butterfly on EEG-only data (the mixed-type version is dominated by the
   synthetic mag/grad amplitudes and shows nothing)
2. the dark-theme axis-label clipping, ragged vs fixed (triage)
3. epoch tick labels when the *short* epoch is at the right edge, and in a
   scrolled window
4. F1's permanent corruption: right, right, left, left
"""

import numpy as np

from vis import app, build, open_qt, rep_spec, shot_qt


def labels(fig):
    return fig.mne.plt.getAxis("bottom").get_labels()


def painted(fig):
    from qtpy.QtGui import QPainter, QPixmap

    ax = fig.mne.plt.getAxis("bottom")
    pm = QPixmap(max(int(ax.width()), 1), max(int(ax.height()), 1))
    p = QPainter(pm)
    try:
        _, _, textSpecs = ax.generateDrawSpecs(p)
    finally:
        p.end()
    return [s for r, f, s in sorted(textSpecs, key=lambda t: t[0].left())]


print("=== 1. butterfly, EEG only ===")
for label, lengths, fixed in [
    ("ragged", (100, 250, 75, 180), False),
    ("fixed", (150, 150, 150, 150), True),
]:
    case = build(rep_spec(lengths, n_channels=20, force_fixed=fixed))
    fig = open_qt(case, n_epochs=4, butterfly=True)
    shot_qt(fig, f"a8_butterfly_eeg_{label}.png")
    vb = fig.mne.viewbox
    x0, x1 = (float(v) for v in vb.viewRange()[0])
    bw = vb.width() or 1
    sep = sorted(
        float(i.value())
        for i in vb.addedItems
        if type(i).__name__ == "InfiniteLine" and i.angle == 90
    )
    print(f"  {label}: butterfly={fig.mne.butterfly} traces={len(fig.mne.traces)} "
          f"separators={np.round(sep, 4).tolist()} labels={labels(fig)}")
    print(f"    separator px = {[round((t - x0) / (x1 - x0) * bw, 1) for t in sep]} "
          f"of {bw:.0f}px")
    fig.close()

print("\n=== 2. dark theme, ragged vs fixed ===")
for label, lengths, fixed in [
    ("ragged", (100, 250, 75, 180), False),
    ("fixed", (150, 150, 150, 150), True),
]:
    case = build(rep_spec(lengths, n_channels=20, force_fixed=fixed))
    for theme in ("light", "dark"):
        fig = open_qt(case, n_epochs=4, theme=theme)
        shot_qt(fig, f"a8_theme_{theme}_{label}.png")
        ax = fig.mne.plt.getAxis("bottom")
        print(f"  {label}/{theme}: mne.dark={fig.mne.dark} "
              f"axis height={ax.height():.0f} labels={labels(fig)} "
              f"painted={painted(fig)}")
        fig.close()

print("\n=== 3. epoch tick labels, short epoch at the right edge / scrolled ===")
for label, lengths, fixed, n_ep in [
    ("short_last", (300, 300, 300, 3), False, 4),
    ("short_first", (3, 300, 300, 300), False, 4),
    ("equal4", (225, 225, 225, 225), True, 4),
    ("scroll_short_middle", (300, 3, 300, 300), False, 2),
]:
    case = build(rep_spec(lengths, n_channels=20, force_fixed=fixed))
    fig = open_qt(case, n_epochs=n_ep)
    for step in range(3 if n_ep == 2 else 1):
        if step:
            fig.hscroll("right")
            app().processEvents()
        w = labels(fig)
        g = painted(fig)
        miss = [s for s in w if s not in g]
        tag = f"{label}_scroll{step}" if n_ep == 2 else label
        print(f"  {tag}: view="
              f"{[round(float(v), 3) for v in fig.mne.viewbox.viewRange()[0]]} "
              f"would say {w} paints {g} missing={miss}")
        if miss:
            shot_qt(fig, f"a8_ticks_{tag}.png")
    fig.close()

print("\n=== 4. F1 permanence: right, right, left, left ===")
for label, lengths, fixed in [
    ("ragged", (100, 250, 75, 180), False),
    ("fixed", (150, 150, 150, 150), True),
]:
    case = build(rep_spec(lengths, n_channels=20, force_fixed=fixed))
    fig = open_qt(case, n_epochs=2)
    fig._add_vline(0.9)
    app().processEvents()
    seq = ["<add 0.9>", "right", "right", "left", "left"]
    for i, step in enumerate(seq):
        if step != "<add 0.9>":
            fig.hscroll(step)
            app().processEvents()
        lat = []
        for vl in fig.mne.vline:
            x = float(vl.value())
            ix = int(np.clip(np.searchsorted(case.boundary_times[1:], x, side="right"),
                             0, case.n_epochs - 1))
            lat.append(round(float(case.tmins[ix] +
                       round((x - case.boundary_times[ix]) * case.sfreq) / case.sfreq), 3))
        vis = [vl.isVisible() for vl in fig.mne.vline]
        print(f"  {label} after {step:>9}: latencies={lat} visible={vis} view="
              f"{[round(float(v), 3) for v in fig.mne.viewbox.viewRange()[0]]}")
        if i == len(seq) - 1:
            shot_qt(fig, f"a8_f1_{label}_4_permanent.png")
    fig.close()
