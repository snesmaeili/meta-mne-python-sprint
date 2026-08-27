"""A3 — the three confirmed defects, one runnable script.

    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \
    PYTHONPATH="D:/mne-python;D:/meta-mne-python-sprint/validation/browser_fuzz" \
    python a3_repro.py
"""

import warnings

import numpy as np

import mne
from a3_lifecycle import banner, make

SFREQ = 100.0


def a3_1_stale_axis():
    banner("A3-1  subselection leaves the common time axis at the parent's span")
    ep = make((100, 250, 75, 180), (0.0, -0.2, 0.1, -0.5))
    print(f"  4 ragged epochs; union axis spans "
          f"[{ep._raw_times[0]:+.2f}, {ep._raw_times[-1]:+.2f}] "
          f"= {len(ep._raw_times)} samples")

    sub = ep[0]  # one 100-sample epoch, tmin 0, tmax 0.99
    print(f"  epochs[0]: 1 epoch, {sub._data[0].shape[-1]} samples, "
          f"tmin={sub.tmin[0]:+.2f} tmax={sub.tmax[0]:+.2f}")
    print(f"             _raw_times STILL [{sub._raw_times[0]:+.2f}, "
          f"{sub._raw_times[-1]:+.2f}] = {len(sub._raw_times)} samples")

    fixed, n_contrib = sub.as_fixed()
    d = fixed.get_data(copy=False)
    print(f"  as_fixed():        {d.shape}, {int(np.isnan(d).sum())} of {d.size} NaN")
    print(f"  n_contributing==0: {int((n_contrib == 0).sum())} of "
          f"{len(n_contrib)} time points  <-- no epoch reaches there at all")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = sub.to_data_frame()
    msg = [str(w.message) for w in caught if "as_fixed" in str(w.message)][0]
    print(f"  to_data_frame():   {len(df)} rows, "
          f"{int(df['EEG000'].isna().sum())} NaN  (the epoch has 100 samples)")
    print(f"  the warning says:  ...{msg[msg.index('padded'):msg.index('Call')]}")
    print("  ^ the warning names the right window; as_fixed used the stale one")

    banner("A3-1b  the same object can no longer be recognised as fixed-duration")
    ep = make((100, 100, 250))
    two = ep[[0, 1]]  # two 100-sample epochs, same tmin: one shared axis
    print(f"  epochs[[0,1]] of (100,100,250): lengths "
          f"{[d.shape[-1] for d in two._data]}, tmin {two.tmin}")
    print(f"  _variable_duration is still {two._variable_duration}")
    try:
        two.times
    except RuntimeError as exc:
        print(f"  epochs.times -> RuntimeError: {str(exc)[:96]}")
    try:
        two.average()
    except NotImplementedError as exc:
        print(f"  epochs.average() -> NotImplementedError: {str(exc)[:80]}")
    ep2 = make((100, 100, 250))
    ep2.crop(0.0, 0.99)
    print(f"  by contrast crop(0, 0.99) DOES re-collapse: "
          f"_variable_duration={ep2._variable_duration}")

    banner("A3-1c  the browser's own close path reaches it")
    ep = make((100, 250, 75, 180), (0.0, -0.2, 0.1, -0.5))
    mne.viz.set_browser_backend("matplotlib")
    fig = ep.plot(n_epochs=2, show=False)
    fig.test_mode = True
    bt = np.concatenate([[0], np.cumsum([100, 250, 75, 180])]) / SFREQ
    for pos in (1, 2, 3):
        while True:
            a, b = fig._get_epoch_ix_range()
            if a <= pos < b:
                break
            fig._fake_keypress("right" if pos >= b else "left")
        x = (bt[pos] + bt[pos + 1]) / 2
        tr = fig.mne.traces[0]
        xd = np.asarray(tr.get_xdata(), float)
        yd = np.asarray(tr.get_ydata(), float)
        ok = np.isfinite(xd) & np.isfinite(yd)
        j = int(np.nanargmin(np.where(ok, np.abs(xd - x), np.inf)))
        fig._fake_click((float(xd[j]), float(yd[j])), xform="data")
    print(f"  marked bad in the browser: {list(fig.mne.bad_epochs)}")
    fig._close_impl()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = len(ep.to_data_frame())
    print(f"  after closing: {len(ep)} epoch of "
          f"{ep._data[0].shape[-1]} samples; to_data_frame gives {rows} rows")
    import matplotlib.pyplot as plt

    plt.close("all")


def a3_2_drop_bad():
    banner("A3-2  drop_bad(reject=) / drop_bad(flat=) leak a RuntimeError")
    for kw in (dict(reject=dict(eeg=200e-6)), dict(flat=dict(eeg=1e-15))):
        ep = make((100, 250, 75, 180, 120))
        try:
            ep.drop_bad(verbose=False, **kw)
            print(f"  drop_bad({kw}) -> ok")
        except Exception as exc:
            import traceback

            fr = traceback.extract_tb(exc.__traceback__)[-1]
            print(f"  drop_bad({list(kw)[0]}=...) -> {type(exc).__name__} at "
                  f"{fr.filename.split(chr(92))[-1]}:{fr.lineno}")
            print(f"      {str(exc)[:110]}")
    print("  but the same rejection AT CONSTRUCTION works:")
    from mne import create_info
    from mne.io import RawArray

    rng = np.random.default_rng(0)
    info = create_info(["a", "b", "c"], SFREQ, "eeg")
    raw = RawArray(rng.standard_normal((3, 3000)) * 1e-6, info, verbose=False)
    raw._data[1, 520] = 500e-6
    ev = np.column_stack(
        [np.array([300, 500, 900, 1400, 2000]), np.zeros(5, int), np.ones(5, int)]
    )
    ep = mne.Epochs(
        raw, ev, event_id={"x": 1}, tmin=np.zeros(5),
        tmax=np.array([0.99, 2.49, 0.74, 1.79, 1.19]), baseline=None,
        preload=True, reject=dict(eeg=100e-6), verbose=False,
    )
    print(f"      Epochs(..., reject=...) kept {len(ep)} of 5, "
          f"selection {list(map(int, ep.selection))}, drop_log {ep.drop_log}")


def a3_3_concatenate():
    banner("A3-3  concatenate_epochs leaks a RuntimeError about `times`")
    a = make((100, 250), (0.0, -0.2))
    b = make((75, 180), (0.1, -0.5))
    try:
        mne.concatenate_epochs([a, b])
    except Exception as exc:
        import traceback

        fr = traceback.extract_tb(exc.__traceback__)[-1]
        print(f"  {type(exc).__name__} at {fr.filename.split(chr(92))[-1]}:{fr.lineno}")
        print(f"      {str(exc)[:150]}")
    print("  the message counts 2 epochs; the user asked to join 4")


if __name__ == "__main__":
    a3_1_stale_axis()
    a3_2_drop_bad()
    a3_3_concatenate()
