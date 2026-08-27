"""A8-4: which epoch-number tick labels the Qt time axis actually paints.

``TimeAxis.get_labels()`` reports what it *would* say; pyqtgraph's
``generateDrawSpecs`` reports what is really painted. Compare the two across
several window widths, ragged vs equal.
"""

import numpy as np
from qtpy.QtGui import QPainter, QPixmap

from vis import app, build, open_qt, rep_spec, shot_qt


def painted_ticks(fig):
    """(tick value, string, x px) for every label the axis really paints."""
    ax = fig.mne.plt.getAxis("bottom")
    pm = QPixmap(max(int(ax.width()), 1), max(int(ax.height()), 1))
    p = QPainter(pm)
    try:
        _, tickSpecs, textSpecs = ax.generateDrawSpecs(p)
    finally:
        p.end()
    out = [(float(r.center().x()), str(s)) for r, flags, s in textSpecs]
    out.sort()
    return out, len(tickSpecs)


CASES = [
    ("hundred_to_one", (3, 300, 3, 300), False),
    ("tiny_adjacent", (3, 3, 300, 300), False),
    ("reference", (100, 250, 75, 180), False),
    ("equal_control", (150, 150, 150, 150), True),
    ("equal_ragged_path", (150, 150, 150, 150), False),
]

for label, lengths, fixed in CASES:
    case = build(rep_spec(lengths, n_channels=20, force_fixed=fixed))
    print(f"\n=== {label} lengths={lengths} boundaries="
          f"{np.round(case.boundary_times, 4).tolist()} ===")
    for w in (1600, 900, 600, 420, 300):
        fig = open_qt(case, n_epochs=4)
        fig.resize(w, 450)
        fig.show()
        app().processEvents()
        app().processEvents()
        ax = fig.mne.plt.getAxis("bottom")
        want = ax.get_labels()
        got, n_ticks = painted_ticks(fig)
        got_txt = [s for _, s in got]
        missing = [s for s in want if s not in got_txt]
        overlaps = []
        for (xa, sa), (xb, sb) in zip(got, got[1:]):
            if xb - xa < 12:
                overlaps.append((sa, sb, round(xb - xa, 1)))
        status = "OK" if not missing and not overlaps else "PROBLEM"
        print(f"  {w:>4}px  would say {want}  paints {got_txt}  "
              f"missing={missing}  overlap<12px={overlaps}  [{status}]")
        if missing or overlaps:
            shot_qt(fig, f"a8_ticks_{label}_{w}px.png", w=w, h=450)
        fig.close()
