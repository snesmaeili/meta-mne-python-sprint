"""A8-1: boundary lines and epoch tick labels at unequal spacing (Qt).

Ragged (3,300,3,300) and (2,3,2,3) against the equal-duration control, at
several window widths, with the tick positions printed next to each shot.
"""

import numpy as np

from vis import build, open_qt, print_ticks, qt_numbers, rep_spec, setx, shot_qt, Spec


def boundary_line_report(fig, case):
    """Where the browser actually drew its epoch separator lines."""
    vb = fig.mne.viewbox
    (x0, x1) = (float(v) for v in vb.viewRange()[0])
    w = vb.width() or 1
    lines = []
    for item in vb.addedItems:
        cls = type(item).__name__
        if cls == "InfiniteLine" and item.angle == 90:
            lines.append(float(item.value()))
    lines.sort()
    print(f"    separator lines drawn at t={np.round(lines, 4).tolist()}")
    want = [float(t) for t in case.boundary_times[1:-1]]
    print(f"    expected (boundary_times[1:-1]) = {np.round(want, 4).tolist()}")
    inview = [t for t in want if x0 - 1e-9 <= t <= x1 + 1e-9]
    got = [t for t in lines if x0 - 1e-9 <= t <= x1 + 1e-9]
    ok = len(inview) == len(got) and all(
        abs(a - b) < 1e-9 for a, b in zip(sorted(inview), sorted(got))
    )
    print(f"    in-view separators match: {ok}  ({len(got)} drawn / {len(inview)} due)")
    px = [(t - x0) / (x1 - x0) * w for t in got]
    print(f"    separator pixel x = {[round(p, 1) for p in px]}  (width {w:.0f}px)")
    return ok


CASES = [
    ("hundred_to_one", (3, 300, 3, 300), False),
    ("equal_control", (150, 150, 150, 150), True),
    ("tiny_adjacent", (3, 3, 300, 300), False),
    ("all_tiny", (2, 3, 2, 3), False),
]

for label, lengths, fixed in CASES:
    print(f"\n=== {label}  lengths={lengths} force_fixed={fixed} ===")
    spec = rep_spec(lengths, n_channels=20, force_fixed=fixed)
    case = build(spec)
    for w in (900, 420, 1600):
        fig = open_qt(case, n_epochs=4)
        shot_qt(fig, f"a8_bounds_{label}_{w}px.png", w=w, h=450)
        qt_numbers(fig, case)
        boundary_line_report(fig, case)
        print_ticks(fig, f"@{w}px")
        fig.close()
