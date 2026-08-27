"""A3 — validate the candidate fix for the stale common time axis by monkeypatch.

No repo file is modified.
"""

import numpy as np

import mne
from mne.utils.mixin import GetEpochsMixin
from a3_lifecycle import banner, make

SFREQ = 100.0
LENGTHS = (100, 250, 75, 180, 120)
TMINS = (0.0, -0.2, 0.1, -0.5, 0.3)

_orig = GetEpochsMixin._getitem


def _patched(self, item, *args, **kwargs):
    out = _orig(self, item, *args, **kwargs)
    inst = out[0] if isinstance(out, tuple) else out
    if getattr(inst, "_variable_duration", False) and len(inst._tmin_per_epoch):
        sfreq = float(inst.info["sfreq"])
        start_idx = int(round(inst._tmin_per_epoch.min() * sfreq))
        stop_idx = int(round(inst._tmax_per_epoch.max() * sfreq))
        inst._raw_times = np.arange(start_idx, stop_idx + 1) / sfreq
        inst._set_times(inst._raw_times)
    return out


def report(tag):
    print(f"\n--- {tag} ---")
    for item, lbl in [(0, "[0]"), ([0, 2], "[0,2]"), ([1], "[1]"), (slice(None), "[:]")]:
        ep = make(LENGTHS, TMINS)
        sub = ep[item]
        f, n = sub.as_fixed()
        want0 = sub._tmin_per_epoch.min()
        want1 = sub._tmax_per_epoch.max()
        want_n = int(round((want1 - want0) * SFREQ)) + 1
        print(
            f"  {lbl:8s} as_fixed {f.get_data(copy=False).shape[-1]:4d} samples "
            f"[{f.times[0]:+.2f},{f.times[-1]:+.2f}]  want {want_n:4d} "
            f"[{want0:+.2f},{want1:+.2f}]  "
            f"n_contributing==0 at {int((np.asarray(n) == 0).sum())} points  "
            f"{'OK' if f.get_data(copy=False).shape[-1] == want_n else 'STALE'}"
        )
    ep = make(LENGTHS, TMINS)
    ep.drop([1, 2, 3, 4], verbose=False)
    print(f"  drop->1  to_data_frame rows {len(ep.to_data_frame())} (epoch has 100)")


def main():
    banner("BEFORE the fix")
    report("branch as-is")
    banner("AFTER the fix (monkeypatched _getitem)")
    GetEpochsMixin._getitem = _patched
    report("with _raw_times re-derived")

    banner("regression check: the fixed path is untouched")
    from mne import EpochsArray, create_info

    info = create_info(["a", "b", "c"], SFREQ, "eeg")
    d = np.random.default_rng(0).standard_normal((5, 3, 100)) * 1e-6
    ev = np.column_stack([np.arange(5) * 300 + 300, np.zeros(5, int), np.ones(5, int)])
    fx = EpochsArray(
        d, info, events=ev, tmin=-0.2, event_id={"x": 1}, baseline=None, verbose=False
    )
    sub = fx[[3, 1]]
    print(
        f"  fixed [[3,1]]: times [{sub.times[0]}, {sub.times[-1]}] n={len(sub.times)} "
        f"data {sub.get_data(copy=False).shape}"
    )

    banner("empty selection must not raise under the fix")
    ep = make(LENGTHS, TMINS)
    e0 = ep[[]]
    print(f"  epochs[[]] -> {len(e0)} epochs, _raw_times n={len(e0._raw_times)}")

    banner("browser after the fix still lands on the right boundaries")
    for backend in ("matplotlib", "qt"):
        ep = make(LENGTHS, TMINS)
        sub = ep[[3, 1]]
        mne.viz.set_browser_backend(backend)
        fig = sub.plot(n_epochs=2, show=False)
        fig.test_mode = True
        want = np.array([0, 180, 430]) / SFREQ
        good = np.allclose(fig.mne.boundary_times, want)
        print(f"  {backend}: boundary_times {fig.mne.boundary_times} "
              f"{'OK' if good else 'WRONG, want ' + str(want)}")
        try:
            fig.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
