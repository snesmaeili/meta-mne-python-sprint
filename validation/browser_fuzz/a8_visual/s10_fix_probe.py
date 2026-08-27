"""A8-10: the dropped epoch-number label -- triage against the fixed path, and
a monkeypatch probe of a candidate fix.

pyqtgraph drops a tick label whose text rect is not fully inside the axis
boundingRect. For a time axis the label is a *time*, so dropping one at the
edge is right. For the epochs axis the label is an epoch *number* -- an
identifier for a region, not a coordinate -- so clamping it inside is better
than dropping it.

The patch is applied to TimeAxis only (patching AxisItem also catches
ChannelAxis, whose tickStrings then indexes past the channel list).
No file in mne-python or mne-qt-browser is modified.
"""

import numpy as np
from qtpy.QtCore import QRectF
from qtpy.QtGui import QPainter, QPixmap

import mne_qt_browser._widgets as W
from vis import app, build, open_qt, rep_spec, shot_qt

_orig = W.AxisItem.generateDrawSpecs


def patched(self, p):
    axisSpec, tickSpecs, textSpecs = _orig(self, p)
    if not self.mne.is_epochs or not textSpecs:
        return axisSpec, tickSpecs, textSpecs
    br = self.boundingRect()
    have = {round(r.center().x(), 3) for r, _, _ in textSpecs}
    minVal, maxVal = self.mne.viewbox.viewRange()[0]
    values = self.tickValues(minVal, maxVal, self.mne.xmax)[0][1]
    strings = self.tickStrings(list(values), 1.0, 0.0)
    proto, flags, _ = textSpecs[0]
    dif = (self.range[1] - self.range[0]) or 1
    xScale = br.width() / dif
    offset = self.range[0] * xScale
    out = list(textSpecs)
    for v, s in zip(values, strings):
        x = v * xScale - offset
        if round(x, 3) in have:
            continue
        w, h = proto.width(), proto.height()
        left = min(max(x - w / 2.0, br.left()), br.right() - w)
        out.append((QRectF(left, proto.top(), w, h), flags, s))
    return axisSpec, tickSpecs, out


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
    ("ragged short_first", (3, 300, 300, 300), False, 4),
    ("ragged short_last", (300, 300, 300, 3), False, 4),
    ("ragged 100:1", (3, 300, 3, 300), False, 4),
    ("ragged tiny_adjacent", (3, 3, 300, 300), False, 4),
    ("ragged reference", (100, 250, 75, 180), False, 4),
    ("FIXED equal 4ep", (150, 150, 150, 150), True, 4),
    ("FIXED equal 50ep", (60,) * 50, True, 50),
    ("FIXED equal 100ep", (60,) * 100, True, 100),
]

for tag, fn in (("BEFORE", None), ("AFTER", patched)):
    if fn is None:
        try:
            del W.TimeAxis.generateDrawSpecs
        except AttributeError:
            pass
    else:
        W.TimeAxis.generateDrawSpecs = fn
    print(f"\n--- {tag} ---")
    for label, lengths, fixed, n_ep in CASES:
        case = build(rep_spec(lengths, n_channels=8, force_fixed=fixed))
        for w in (1600, 900, 420):
            fig = open_qt(case, n_epochs=n_ep)
            fig.resize(w, 450)
            fig.show()
            app().processEvents()
            app().processEvents()
            ax = fig.mne.plt.getAxis("bottom")
            want, got = ax.get_labels(), painted(fig)
            missing = [s for s in want if s not in got]
            mark = "" if not missing else "   <-- DROPPED"
            print(f"  {label:<22}{w:>5}px  want={len(want):<3} painted={len(got):<3} "
                  f"missing={missing}{mark}")
            if tag == "AFTER" and label.startswith("ragged short"):
                shot_qt(fig, f"a8_fixprobe_{label.split()[1]}_{w}px.png", w=w, h=450)
            fig.close()

try:
    del W.TimeAxis.generateDrawSpecs
except AttributeError:
    pass
