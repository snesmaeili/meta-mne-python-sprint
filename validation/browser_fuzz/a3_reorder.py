"""A3-4 — browse a reordered / subselected ragged object, then mark and close."""

import numpy as np

import mne
from a3_lifecycle import audit, banner, lengths_of, make, tags

SFREQ = 100.0
LENGTHS = (100, 250, 75, 180, 120)
FAILS = []


def fail(m):
    FAILS.append(m)
    print("  FAIL", m)


def bounds(lengths):
    bs = np.concatenate([[0], np.cumsum(lengths)]).astype(int)
    return bs, bs / SFREQ


def mark(fig, backend, x):
    if backend == "matplotlib":
        tr = fig.mne.traces[0]
        xd = np.asarray(tr.get_xdata(), float)
        yd = np.asarray(tr.get_ydata(), float)
        okm = np.isfinite(xd) & np.isfinite(yd)
        j = int(np.nanargmin(np.where(okm, np.abs(xd - x), np.inf)))
        fig._fake_click((float(xd[j]), float(yd[j])), xform="data")
    else:
        fig.mne.traces[0].toggle_bad(x)


def one(backend, item, label, mark_pos=(0,)):
    ep = make(LENGTHS)
    sub = ep[item]
    order = tags(sub)
    lengths = lengths_of(sub)
    bs, bt = bounds(lengths)
    sel = list(sub.selection)
    print(f"\n-- {label} [{backend}] order={order} lens={lengths} sel={sel}")

    mne.viz.set_browser_backend(backend)
    fig = sub.plot(n_epochs=min(2, len(sub)), show=False)
    fig.test_mode = True
    msgs = []
    if not np.allclose(fig.mne.boundary_times, bt):
        msgs.append(f"boundary_times {fig.mne.boundary_times} != {bt}")
    et = np.asarray(getattr(fig.mne, "epoch_tmins", []))
    if len(et) and not np.allclose(et, sub._tmin_per_epoch):
        msgs.append(f"epoch_tmins {et} != {sub._tmin_per_epoch}")
    # the epoch axis must name the epochs by their selection number, in the
    # object's own order, not sorted
    for ix in range(len(lengths)):
        x = (bt[ix] + bt[ix + 1]) / 2
        got = fig._get_epoch_num_from_time(x)
        if got != sel[ix]:
            msgs.append(f"epoch at position {ix} reported as {got}, want {sel[ix]}")
    # the window must hold that epoch's own samples
    ix0, ix1 = fig._get_epoch_ix_range()
    start, stop = fig._get_start_stop()
    data, _ = fig._load_data(start, stop)
    want = np.concatenate([sub._data[i] for i in range(ix0, ix1)], axis=-1)
    if data.shape != want.shape or not np.array_equal(data, want):
        msgs.append(f"loaded window {data.shape} != source {want.shape} / differs")

    marked = []
    for pos in mark_pos:
        if pos >= len(lengths):
            continue
        # bring it into view
        guard = 0
        while guard < 40:
            a, b = fig._get_epoch_ix_range()
            if a <= pos < b:
                break
            if backend == "matplotlib":
                fig._fake_keypress("right" if pos >= b else "left")
            else:
                fig.hscroll("right" if pos >= b else "left")
            guard += 1
        mark(fig, backend, (bt[pos] + bt[pos + 1]) / 2)
        marked.append(sel[pos])
    if sorted(fig.mne.bad_epochs) != sorted(marked):
        msgs.append(f"bad_epochs {sorted(fig.mne.bad_epochs)} != {sorted(marked)}")
    fig._close_impl()
    survivors = [p for p in range(len(lengths)) if p not in set(mark_pos)]
    msgs += audit(sub, label, expect_tags=[order[p] for p in survivors])
    if [int(s) for s in sub.selection] != [int(sel[p]) for p in survivors]:
        msgs.append(
            f"selection after close {list(sub.selection)} != "
            f"{[sel[p] for p in survivors]}"
        )
    for m in msgs:
        fail(f"{label}: {m}")
    if not msgs:
        print(
            f"   ok: tags={tags(sub)} lens={lengths_of(sub)} sel={list(sub.selection)}"
        )
    try:
        fig.close()
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass


def main():
    for backend in ("matplotlib", "qt"):
        banner(f"{backend}: browse a reordered / subselected object")
        one(backend, slice(None, None, -1), "reversed [::-1]", (0, 3))
        one(backend, [3, 1], "list [3,1]", (0,))
        one(backend, [4, 0, 2], "list [4,0,2]", (1,))
        one(backend, slice(None, None, 2), "[::2]", (2,))
        one(backend, np.array([False, True, True, False, True]), "bool mask", (0, 2))
        one(backend, [2], "single [2]", (0,))
        one(backend, [4, 4, 0], "duplicated [4,4,0]", (0,))
    banner("SUMMARY")
    print(f"{len(FAILS)} violations")
    for f in FAILS:
        print("  -", f)


if __name__ == "__main__":
    main()
