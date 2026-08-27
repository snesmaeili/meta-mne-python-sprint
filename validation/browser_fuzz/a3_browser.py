"""A3-2 — drop lifecycle through the browser: mark bad, close, re-check.

Starts from an already non-contiguous ``selection`` so a marker that reports by
position instead of by selection number is caught.
"""

import sys

import numpy as np

import mne
from a3_lifecycle import audit, banner, lengths_of, make, tags

LENGTHS = (100, 250, 75, 180, 120, 90, 200)
SFREQ = 100.0

FAILS = []


def fail(msg):
    FAILS.append(msg)
    print("  FAIL", msg)


def boundaries(lengths, sfreq=SFREQ):
    bs = np.concatenate([[0], np.cumsum(lengths)]).astype(int)
    return bs, bs / sfreq


def open_fig(ep, backend, **kw):
    mne.viz.set_browser_backend(backend)
    kwargs = dict(n_epochs=2, show=False)
    kwargs.update(kw)
    fig = ep.plot(**kwargs)
    fig.test_mode = True
    return fig


def mark_bad(fig, backend, epoch_ix, bt):
    """Mark the epoch at *position* ``epoch_ix`` bad through the real gesture."""
    x = (bt[epoch_ix] + bt[epoch_ix + 1]) / 2
    if backend == "matplotlib":
        # land the click exactly on a drawn sample of trace 0, the way a real
        # click on the line does; ``line.contains`` is what gates the toggle
        tr = fig.mne.traces[0]
        xd = np.asarray(tr.get_xdata(), float)
        yd = np.asarray(tr.get_ydata(), float)
        ok = np.isfinite(xd) & np.isfinite(yd)
        j = int(np.nanargmin(np.where(ok, np.abs(xd - x), np.inf)))
        fig._fake_click((float(xd[j]), float(yd[j])), xform="data")
    else:
        # exactly what DataTrace.mouseClickEvent does on a left click
        fig.mne.traces[0].toggle_bad(x)


def scroll_to(fig, backend, target_ix):
    """Bring epoch position ``target_ix`` into view."""
    guard = 0
    while guard < 60:
        ix0, ix1 = fig._get_epoch_ix_range()
        if ix0 <= target_ix < ix1:
            return True
        if target_ix >= ix1:
            if backend == "matplotlib":
                fig._fake_keypress("right")
            else:
                fig.hscroll("right")
        else:
            if backend == "matplotlib":
                fig._fake_keypress("left")
            else:
                fig.hscroll("left")
        guard += 1
    return False


def case(backend, drop_first, mark_positions, label, n_epochs=2):
    ep = make(LENGTHS)
    if drop_first:
        ep.drop(list(drop_first), verbose=False)
    kept = [i for i in range(len(LENGTHS)) if i not in set(drop_first or ())]
    lengths = [LENGTHS[i] for i in kept]
    bs, bt = boundaries(lengths)
    sel_before = list(ep.selection)
    print(f"\n-- {label} [{backend}] selection={sel_before} lens={lengths}")

    fig = open_fig(ep, backend, n_epochs=n_epochs)
    # the browser's own model must match the arrays we computed
    if not np.allclose(fig.mne.boundary_times, bt):
        fail(f"{label}: boundary_times {fig.mne.boundary_times} != {bt}")

    marked_nums = []
    for pos in mark_positions:
        if not scroll_to(fig, backend, pos):
            fail(f"{label}: could not scroll epoch {pos} into view")
            continue
        x = (bt[pos] + bt[pos + 1]) / 2
        num = fig._get_epoch_num_from_time(x)
        want_num = sel_before[pos]
        if num != want_num:
            fail(
                f"{label}: _get_epoch_num_from_time at position {pos} "
                f"reported {num}, selection says {want_num}"
            )
        mark_bad(fig, backend, pos, bt)
        marked_nums.append(want_num)

    got = sorted(fig.mne.bad_epochs)
    if got != sorted(marked_nums):
        fail(f"{label}: bad_epochs {got} != marked selection numbers "
             f"{sorted(marked_nums)}")
    else:
        print(f"   bad_epochs = {got}  (selection numbers, correct)")

    fig._close_impl()

    survivors = [p for p in range(len(kept)) if p not in set(mark_positions)]
    expect_tags = [kept[p] for p in survivors]
    msgs = audit(ep, label, expect_tags=expect_tags)
    for m in msgs:
        fail(m)
    if not msgs:
        print(
            f"   after close: tags={tags(ep)} lens={lengths_of(ep)} "
            f"sel={list(ep.selection)} n_tmin={len(ep._tmin_per_epoch)}"
        )
    exp_sel = [sel_before[p] for p in survivors]
    if list(ep.selection) != exp_sel:
        fail(f"{label}: selection after close {list(ep.selection)} != {exp_sel}")
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass


def main():
    backends = sys.argv[1:] or ["matplotlib", "qt"]
    for backend in backends:
        banner(f"{backend}: drop() before plot, mark in browser, close")
        case(backend, (1, 3), [0], "drop(1,3) mark pos0")
        case(backend, (1, 3), [1, 3], "drop(1,3) mark pos1+3")
        case(backend, (1, 3), [4], "drop(1,3) mark last")
        case(backend, (0, 2, 4), [0, 1, 2, 3], "drop(0,2,4) mark ALL")
        case(backend, (), [0, 1, 2, 3, 4, 5, 6], "mark every epoch")
        case(backend, (1, 3), [2], "drop(1,3) mark shortest(75->pos1?)")
        case(backend, (), [], "no marks at all")
        # shortest epoch is source 2 (75 samples); after drop(1,3) it sits at
        # position 1
        case(backend, (1, 3), [1], "drop(1,3) mark the shortest")
        case(backend, (), [6], "mark only the last")
        case(backend, (), [2], "mark only the shortest")

    banner("SUMMARY")
    print(f"{len(FAILS)} violations")
    for f in FAILS:
        print("  -", f)


if __name__ == "__main__":
    main()
