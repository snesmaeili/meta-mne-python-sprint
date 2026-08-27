"""``events=True`` on ragged epochs: per-epoch tmin, overlapping epochs, and
multiple ``event_id`` codes.

Builds its own epochs (the harness generator deliberately spaces events so they
cannot overlap) and computes the expected event-line positions from the event
array and the per-epoch bounds, never from the browser.
"""

import numpy as np

import mne

from .common import assert_env, close_fig


def make(lengths, tmins, onsets, codes, *, sfreq=100.0, n_ch=3, fixed=False):
    lengths = np.asarray(lengths, int)
    tmins = np.asarray(tmins, float)
    onsets = np.asarray(onsets, int)
    codes = np.asarray(codes, int)
    rng = np.random.default_rng(0)
    names = [f"EEG{i:03d}" for i in range(n_ch)]
    info = mne.create_info(names, sfreq, ["eeg"] * n_ch)
    events = np.column_stack([onsets, np.zeros(len(onsets), int), codes])
    eid = {f"e{c}": int(c) for c in sorted(set(codes.tolist()))}
    if fixed:
        assert len(set(lengths.tolist())) == 1
        data = rng.standard_normal((len(lengths), n_ch, lengths[0])) * 1e-6
        ep = mne.EpochsArray(
            data, info, events=events, tmin=float(tmins[0]), event_id=eid,
            baseline=None, verbose=False,
        )
    else:
        arrays = [rng.standard_normal((n_ch, n)) * 1e-6 for n in lengths]
        ep = mne.EpochsArray(
            arrays, info, events=events, tmin=tmins, event_id=eid,
            baseline=None, verbose=False,
        )
    bs = np.concatenate([[0], np.cumsum(lengths)])
    return ep, bs, bs / sfreq, lengths, tmins, events, sfreq


def expected_lines(bs_t, lengths, tmins, events, sfreq):
    """(x, code) for every event line the picture should carry.

    An event belongs to every epoch whose own samples contain it; its x is the
    epoch's boundary plus the event's offset inside that epoch.
    """
    first = events[:, 0] - np.round(-tmins * sfreq).astype(int)
    last = first + lengths - 1
    out = []
    for samp, code in zip(events[:, 0], events[:, 2]):
        for k in range(len(lengths)):
            if first[k] <= samp <= last[k]:
                out.append((bs_t[k] + (samp - first[k]) / sfreq, int(code)))
    return sorted(out)


def drawn_lines(fig):
    lc = fig.mne.event_lines
    if lc is None or not hasattr(lc, "get_segments"):
        return None
    segs = lc.get_segments()
    xs = [float(s[0, 0]) for s in segs]
    texts = [t.get_text() for t in fig.mne.event_texts]
    return sorted(zip(xs, texts))


def probe(label, lengths, tmins, onsets, codes, *, fixed=False, n_epochs=None):
    ep, bs, bs_t, lengths, tmins, events, sfreq = make(
        lengths, tmins, onsets, codes, fixed=fixed
    )
    n_epochs = n_epochs or len(lengths)
    mne.viz.set_browser_backend("matplotlib")
    fig = ep.plot(n_epochs=n_epochs, events=True, show=False)
    fig.test_mode = True
    start, stop = fig._get_start_stop()
    lo, hi = start / sfreq, (stop - 1) / sfreq
    want = [(x, c) for x, c in expected_lines(bs_t, lengths, tmins, events, sfreq)
            if lo - 1e-9 <= x <= hi + 1e-9]
    got = drawn_lines(fig)
    print(f"\n--- {label}")
    print(f"    lengths={lengths.tolist()} tmins={tmins.tolist()} boundaries={np.round(bs_t, 4).tolist()}")
    print(f"    events={events[:, [0, 2]].tolist()}  window samples [{start}, {stop})")
    print(f"    expected x={[round(x, 4) for x, _ in want]} codes={[c for _, c in want]}")
    print(f"    drawn    x={[round(x, 4) for x, _ in got]} labels={[t for _, t in got]}")
    ok = len(got) == len(want) and all(
        abs(g[0] - w[0]) < 1e-9 and g[1] == str(w[1]) for g, w in zip(got, want)
    )
    print(f"    {'OK' if ok else 'MISMATCH  <---'}")
    # also check total (unmasked) model, so a wrong x that is merely off-screen
    # still shows up
    all_want = expected_lines(bs_t, lengths, tmins, events, sfreq)
    model = sorted(zip(np.asarray(fig.mne.event_times, float).tolist(),
                       np.asarray(fig.mne.event_nums, int).tolist()))
    ok2 = len(model) == len(all_want) and all(
        abs(a[0] - b[0]) < 1e-9 and a[1] == b[1] for a, b in zip(model, all_want)
    )
    if not ok2:
        print(f"    MODEL MISMATCH: event_times={[round(x, 4) for x, _ in model]} "
              f"nums={[c for _, c in model]}")
        print(f"                    expected     ={[round(x, 4) for x, _ in all_want]} "
              f"nums={[c for _, c in all_want]}")
    close_fig(fig)
    return ok and ok2


def main():
    assert_env()
    mne.set_log_level("error")
    ok = []

    # 1. plain ragged, tmin 0
    ok.append(probe("ragged tmin=0", [100, 250, 75, 180], [0, 0, 0, 0],
                    [500, 1000, 1500, 2000], [1, 1, 1, 1]))
    # 2. per-epoch tmin (the A8 configuration, re-verified numerically)
    ok.append(probe("ragged per-epoch tmin", [100, 250, 75, 180],
                    [0.0, -0.2, 0.1, -0.5], [500, 1000, 1500, 2000], [1, 1, 1, 1]))
    # 3. multiple event ids
    ok.append(probe("ragged multi event_id", [100, 250, 75, 180], [0, 0, 0, 0],
                    [500, 1000, 1500, 2000], [1, 2, 3, 2]))
    # 4. multi id + per-epoch tmin
    ok.append(probe("ragged multi id + tmin", [100, 250, 75, 180],
                    [-0.3, 0.0, -0.1, 0.2], [500, 1000, 1500, 2000], [1, 2, 3, 2]))
    # 5. OVERLAPPING epochs: onsets closer together than the epoch lengths, so
    #    one event falls inside several epochs and must be drawn several times
    ok.append(probe("ragged overlapping", [300, 300, 300], [0, 0, 0],
                    [500, 600, 700], [1, 2, 3]))
    ok.append(probe("ragged overlapping ragged lengths", [300, 50, 300],
                    [0, 0, 0], [500, 600, 700], [1, 2, 3]))
    ok.append(probe("ragged overlapping + tmin", [300, 50, 300],
                    [-0.5, 0.0, 0.25], [500, 600, 700], [1, 2, 3]))
    # 6. fixed controls
    ok.append(probe("FIXED tmin=0", [100, 100, 100, 100], [0, 0, 0, 0],
                    [500, 1000, 1500, 2000], [1, 1, 1, 1], fixed=True))
    ok.append(probe("FIXED tmin=-0.2", [100, 100, 100, 100],
                    [-0.2, -0.2, -0.2, -0.2], [500, 1000, 1500, 2000],
                    [1, 1, 1, 1], fixed=True))
    ok.append(probe("FIXED overlapping", [300, 300, 300], [0, 0, 0],
                    [500, 600, 700], [1, 2, 3], fixed=True))
    ok.append(probe("FIXED overlapping tmin=-0.5", [300, 300, 300],
                    [-0.5, -0.5, -0.5], [500, 600, 700], [1, 2, 3], fixed=True))
    # 7. scrolled window
    ok.append(probe("ragged overlapping, n_epochs=1", [300, 50, 300], [0, 0, 0],
                    [500, 600, 700], [1, 2, 3], n_epochs=1))

    print(f"\n{sum(ok)}/{len(ok)} configurations matched")


if __name__ == "__main__":
    main()
