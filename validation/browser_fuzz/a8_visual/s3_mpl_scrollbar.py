"""A8-3: the matplotlib horizontal scrollbar, one Rectangle per epoch.

Does each epoch get its own width, is the bar legible with 50+ epochs, and do
the epoch-number labels on it land on the right epochs?
"""

import types

import numpy as np

from vis import build, open_mpl, rep_spec, shot_mpl, Spec


def hscroll_report(fig, case, tag):
    ax = fig.mne.ax_hscroll
    pats = ax.patches
    print(f"  hscroll {tag}: {len(pats)} patches for {case.n_epochs} epochs")
    xs = np.array([p.get_x() for p in pats[: case.n_epochs]])
    ws = np.array([p.get_width() for p in pats[: case.n_epochs]])
    want_x = case.boundary_times[:-1]
    want_w = np.diff(case.boundary_times)
    print(f"    x    max|err| = {np.abs(xs - want_x).max():.3e}")
    print(f"    width max|err| = {np.abs(ws - want_w).max():.3e}")
    # pixel widths of the drawn rectangles
    x0, x1 = ax.get_xlim()
    bb = ax.get_window_extent()
    px = ws / (x1 - x0) * bb.width
    print(
        f"    axes width {bb.width:.0f}px -> epoch pixel widths "
        f"min={px.min():.2f} median={np.median(px):.2f} max={px.max():.2f}"
    )
    print(f"    sub-pixel epochs (<1px): {(px < 1).sum()} / {len(px)}")
    # the minor tick labels on the scrollbar
    labs = [
        (float(t.get_loc()), t.label1.get_text())
        for t in ax.xaxis.get_minor_ticks()
        if t.label1.get_text()
    ]
    ok = True
    for loc, txt in labs:
        ix = int(np.searchsorted(case.boundary_times[1:], loc, side="right"))
        ix = min(ix, case.n_epochs - 1)
        want = str(fig.mne.inst.selection[ix])
        if txt != want:
            ok = False
            print(f"    LABEL MISMATCH at t={loc:.4f}: reads {txt!r}, epoch is {want!r}")
    print(f"    {len(labs)} scrollbar epoch labels, all naming the right epoch: {ok}")
    return ok


def main_axis_report(fig, case, tag):
    ax = fig.mne.ax_main
    x0, x1 = ax.get_xlim()
    bb = ax.get_window_extent()
    minor = [
        (float(t.get_loc()), t.label1.get_text())
        for t in ax.xaxis.get_minor_ticks()
        if t.label1.get_text() and x0 <= t.get_loc() <= x1
    ]
    minor.sort()
    print(f"  main axis {tag}: xlim=[{x0:.4f}, {x1:.4f}] {bb.width:.0f}px")
    prev = None
    for loc, txt in minor:
        p = (loc - x0) / (x1 - x0) * bb.width
        ix = int(np.searchsorted(case.boundary_times[1:], loc, side="right"))
        ix = min(ix, case.n_epochs - 1)
        want = str(fig.mne.inst.selection[ix])
        gap = "" if prev is None else f" gap={p - prev:6.1f}px"
        flag = "" if txt == want else f"  <-- reads {txt!r}, epoch is {want!r}"
        crowd = "  <-- CROWDED" if prev is not None and p - prev < 14 else ""
        print(f"    t={loc:8.4f} '{txt}' x={p:7.1f}px{gap}{flag}{crowd}")
        prev = p
    gl = [float(ln.get_xdata()[0]) for ln in ax.get_xgridlines()]
    print(f"    gridlines (all) = {np.round(sorted(gl), 4).tolist()[:12]}")
    print(f"    boundary_times[1:-1] = {np.round(case.boundary_times[1:-1], 4).tolist()[:12]}")


def mark_bad(fig, t):
    fig._toggle_bad_epoch(types.SimpleNamespace(xdata=float(t)))


# --- 50 epochs, ragged vs equal -------------------------------------------
rag = tuple(20 + (i * 37) % 200 for i in range(50))
eq = tuple([int(round(np.mean(rag)))] * 50)
for label, lengths, fixed in [("ragged50", rag, False), ("equal50", eq, True)]:
    print(f"\n=== mpl scrollbar {label}: {len(lengths)} epochs, "
          f"lengths {lengths[:6]}... total {sum(lengths)} ===")
    case = build(rep_spec(lengths, n_channels=20, force_fixed=fixed))
    fig = open_mpl(case, n_epochs=5)
    # mark a spread of epochs bad so the rectangles become visible
    for ix in (0, 1, 7, 8, 20, 33, 48, 49):
        mark_bad(fig, (case.boundary_times[ix] + case.boundary_times[ix + 1]) / 2)
    shot_mpl(fig, f"a8_mplscroll_{label}.png", w=9.0, h=4.5)
    hscroll_report(fig, case, label)
    print(f"    marked-bad epochs -> durations "
          f"{[round(float(np.diff(case.boundary_times)[i]), 3) for i in (0,1,7,8,20,33,48,49)]}")
    import matplotlib.pyplot as _plt; _plt.close(fig)

# --- the boundary-spacing comparison on the main axis ----------------------
for label, lengths, fixed in [
    ("hundred_to_one", (3, 300, 3, 300), False),
    ("equal_control", (150, 150, 150, 150), True),
    ("reference", (100, 250, 75, 180), False),
]:
    print(f"\n=== mpl main axis {label} lengths={lengths} ===")
    case = build(rep_spec(lengths, n_channels=20, force_fixed=fixed))
    fig = open_mpl(case, n_epochs=4)
    shot_mpl(fig, f"a8_mplbounds_{label}.png", w=9.0, h=4.5)
    main_axis_report(fig, case, label)
    hscroll_report(fig, case, label)
    import matplotlib.pyplot as _plt; _plt.close(fig)
