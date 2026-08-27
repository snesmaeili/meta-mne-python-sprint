"""Self-contained base-commit check for the decim phase (no harness imports).

Runs under ``PYTHONPATH=D:/tmp/mne-base`` as well as under the branch, so the
same numbers can be compared directly. Equal-duration epochs only, because the
base commit cannot build ragged ones.
"""

import numpy as np

import mne


def phases(L, n_ep, decim, n_channels=3, sfreq=100.0, n_epochs_window=4):
    rng = np.random.default_rng(0)
    names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(names, sfreq, ["eeg"] * n_channels)
    data = rng.standard_normal((n_ep, n_channels, L)) * 1e-6
    stride = L + int(sfreq) + 100
    events = np.column_stack(
        [np.arange(n_ep) * stride + stride, np.zeros(n_ep, int), np.ones(n_ep, int)]
    )
    ep = mne.EpochsArray(
        data, info, events=events, tmin=0.0, event_id={"x": 1},
        baseline=None, verbose=False,
    )
    mne.viz.set_browser_backend("matplotlib")
    fig = ep.plot(n_epochs=n_epochs_window, decim=decim, show=False)
    line = fig.mne.traces[0]
    x = np.asarray(np.ma.getdata(line.get_xdata()), float)
    ft = float(fig.mne.first_time)
    samples = np.round((x - ft) * sfreq).astype(int)
    bs = np.arange(n_ep + 1) * L
    out = {}
    for k in range(n_ep):
        inside = samples[(samples >= bs[k]) & (samples < bs[k + 1])]
        if len(inside):
            out[k] = int(inside[0] - bs[k])
    import matplotlib.pyplot as plt

    plt.close("all")
    return out, len(samples)


if __name__ == "__main__":
    print("mne:", mne.__file__, mne.__version__)
    print("variable_duration attr:", hasattr(mne.EpochsArray, "variable_duration"))
    for L in (100, 101):
        for d in (1, 2, 4, 8):
            ph, n = phases(L, 4, d)
            drift = [k for k, v in ph.items() if v != 0]
            print(
                f"L={L:4d} decim={d}: phases={ph} n_drawn={n} "
                f"drifting_epochs={drift}"
            )
