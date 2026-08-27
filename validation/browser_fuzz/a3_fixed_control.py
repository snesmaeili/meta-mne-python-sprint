"""A3-5 — fixed-duration controls for every A3 candidate finding.

Uses only the ordinary 3-D path, so it runs unchanged against the base commit.
"""

import warnings

import numpy as np

import mne
from mne import EpochsArray, create_info

SFREQ = 100.0
N = 5
L = 100


def fixed(n=N, length=L, tmin=0.0, n_channels=3):
    info = create_info([f"EEG{i:03d}" for i in range(n_channels)], SFREQ, ["eeg"] * n_channels)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, n_channels, length)) * 1e-6
    for i in range(n):
        data[i, 0, :] = i * 1e-6
    stride = length + int(SFREQ) + 100
    ev = np.column_stack(
        [np.arange(n) * stride + stride, np.zeros(n, int), np.ones(n, int)]
    )
    return EpochsArray(
        data, info, events=ev, tmin=tmin, event_id={"x": 1}, baseline=None,
        verbose=False,
    )


def line(tag, msg):
    print(f"[{tag}] {msg}")


print("mne:", mne.__file__)

# ---- D1 control: does subselection leave a stale common time axis? ----------
ep = fixed()
sub = ep[0]
line("D1", f"fixed epochs[0].times span [{sub.times[0]}, {sub.times[-1]}] "
     f"n={len(sub.times)}  (source epoch is {L} samples)")
if hasattr(sub, "as_fixed"):
    f, n = sub.as_fixed()
    line("D1", f"as_fixed shape {f.get_data(copy=False).shape}, "
         f"n_contributing zeros {int((np.asarray(n) == 0).sum())}")
line("D1", f"to_data_frame rows {len(sub.to_data_frame())}")
line("D1", f"average() -> {sub.average().data.shape}")

# ---- D2 control: drop_bad(reject=) after preload ---------------------------
ep = fixed()
ep._data[1, 1, 0] = 500e-6
ep._data[3, 1, 0] = 500e-6
try:
    ep.drop_bad(reject=dict(eeg=100e-6), verbose=False)
    line("D2", f"fixed drop_bad(reject=) ok: kept {len(ep)} of {N}, "
         f"selection {list(ep.selection)}")
except Exception as exc:
    line("D2", f"fixed drop_bad RAISED {type(exc).__name__}: {exc}")

# ---- D3 control: concatenate_epochs ----------------------------------------
a, b = fixed(n=2), fixed(n=2)
try:
    out = mne.concatenate_epochs([a, b])
    line("D3", f"fixed concatenate_epochs ok: {len(out)} epochs, "
         f"shape {out.get_data(copy=False).shape}")
except Exception as exc:
    line("D3", f"fixed concatenate_epochs RAISED {type(exc).__name__}: {exc}")

# ---- D4 control: duplicated selection through the browser ------------------
for backend in ("matplotlib", "qt"):
    ep = fixed()
    sub = ep[[4, 4, 0]]
    line("D4", f"{backend}: selection after epochs[[4,4,0]] = {list(sub.selection)}")
    mne.viz.set_browser_backend(backend)
    fig = sub.plot(n_epochs=2, show=False)
    fig.test_mode = True
    bt = np.arange(4) * (L / SFREQ)
    x = (bt[0] + bt[1]) / 2
    if backend == "matplotlib":
        tr = fig.mne.traces[0]
        xd = np.asarray(tr.get_xdata(), float)
        yd = np.asarray(tr.get_ydata(), float)
        ok = np.isfinite(xd) & np.isfinite(yd)
        j = int(np.nanargmin(np.where(ok, np.abs(xd - x), np.inf)))
        fig._fake_click((float(xd[j]), float(yd[j])), xform="data")
    else:
        fig.mne.traces[0].toggle_bad(x)
    line("D4", f"{backend}: bad_epochs {list(fig.mne.bad_epochs)}")
    fig._close_impl()
    line("D4", f"{backend}: after close, {len(sub)} epochs remain, "
         f"selection {list(sub.selection)}  (marked ONE of two numbered 4)")
    try:
        fig.close()
    except Exception:
        pass
    import matplotlib.pyplot as plt

    plt.close("all")

# ---- D5 control: shift_time(relative=False) --------------------------------
ep = fixed(tmin=-0.2)
ep.shift_time(1.0, relative=False)
line("D5", f"fixed shift_time(1.0, relative=False): times "
     f"[{ep.times[0]:.4f}, {ep.times[-1]:.4f}]")
