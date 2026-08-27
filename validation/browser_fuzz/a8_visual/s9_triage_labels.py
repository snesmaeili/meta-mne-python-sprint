"""A8-9 triage: is the dropped epoch-number label ragged-only?

The pyqtgraph cull is a rect-containment test, not new code. The question is
whether the *fixed* path can reach it too. With many equal epochs in one window
the first midpoint also sits close to the edge, so test that explicitly.
"""

import numpy as np
from qtpy.QtGui import QPainter, QPixmap

from vis import app, build, open_qt, rep_spec


def painted(fig):
    ax = fig.mne.plt.getAxis("bottom")
    pm = QPixmap(max(int(ax.width()), 1), max(int(ax.height()), 1))
    p = QPainter(pm)
    try:
        _, _, textSpecs = ax.generateDrawSpecs(p)
    finally:
        p.end()
    return [s for r, f, s in sorted(textSpecs, key=lambda t: t[0].left())]


CASES = [
    ("equal_50ep_fixed", (60,) * 50, True, 50),
    ("equal_50ep_listpath", (60,) * 50, False, 50),
    ("equal_20ep_fixed", (60,) * 20, True, 20),
    ("equal_8ep_fixed", (60,) * 8, True, 8),
    ("equal_4ep_fixed", (60,) * 4, True, 4),
    ("ragged_4ep_short_first", (3, 300, 300, 300), False, 4),
]

for label, lengths, fixed, n_ep in CASES:
    case = build(rep_spec(lengths, n_channels=8, force_fixed=fixed))
    for w in (900, 1600):
        fig = open_qt(case, n_epochs=n_ep)
        fig.resize(w, 450)
        fig.show()
        app().processEvents()
        app().processEvents()
        ax = fig.mne.plt.getAxis("bottom")
        want = ax.get_labels()
        got = painted(fig)
        missing = [s for s in want if s not in got]
        vb = fig.mne.viewbox
        print(
            f"  {label:<24} n_epochs={n_ep:<3} {w:>4}px vb={vb.width():.0f}px  "
            f"get_labels()={len(want)}  painted={len(got)}  missing={missing}"
        )
        fig.close()
