import numpy as np
from qtpy.QtGui import QPainter, QPixmap

from vis import app, build, open_qt, rep_spec


def specs(fig):
    ax = fig.mne.plt.getAxis("bottom")
    pm = QPixmap(max(int(ax.width()), 1), max(int(ax.height()), 1))
    p = QPainter(pm)
    try:
        _, tickSpecs, textSpecs = ax.generateDrawSpecs(p)
    finally:
        p.end()
    return ax, tickSpecs, textSpecs


for label, lengths, fixed in [
    ("hundred_to_one", (3, 300, 3, 300), False),
    ("tiny_adjacent", (3, 3, 300, 300), False),
]:
    case = build(rep_spec(lengths, n_channels=20, force_fixed=fixed))
    for w in (1600, 900):
        fig = open_qt(case, n_epochs=4)
        fig.resize(w, 450)
        fig.show()
        app().processEvents()
        app().processEvents()
        ax, tickSpecs, textSpecs = specs(fig)
        br = ax.boundingRect()
        print(f"{label} @{w}px  axis boundingRect x=[{br.left():.1f},{br.right():.1f}]")
        print(f"   tick line x = {sorted(round(float(a.x()), 1) for _, a, _ in tickSpecs)}")
        for r, flags, s in sorted(textSpecs, key=lambda t: t[0].left()):
            print(
                f"   text {s!r} rect x=[{r.left():.1f},{r.right():.1f}] "
                f"center={r.center().x():.1f}"
            )
        fig.close()
