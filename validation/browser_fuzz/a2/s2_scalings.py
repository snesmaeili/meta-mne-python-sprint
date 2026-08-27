"""S8 -- ``scalings`` on ragged epochs: the auto path, the subset path, the dict
path and the whitening path."""

import numpy as np

import mne
from mne.viz.utils import _compute_scalings

from ..build import Spec
from .common import assert_env, build, close_fig, open_browser


def expected_iqr(case, ch_ixs):
    """IQR of the concatenated source data for the given channel indices."""
    strip = np.concatenate([a[ch_ixs] for a in case.source], axis=-1).ravel()
    return float(np.diff(np.percentile(strip, [25, 75]))[0])


def report_auto(spec, label):
    case = build(spec)
    ep = case.epochs
    got = _compute_scalings("auto", ep.copy())
    ch_types = np.array(ep.get_channel_types())
    print(f"\n--- {label}: lengths={list(case.lengths)}")
    for t in sorted(set(ch_types)):
        ixs = np.flatnonzero(ch_types == t)
        want = expected_iqr(case, ixs)
        g = got.get(t)
        if g is None or isinstance(g, str):
            print(f"    {t}: browser={g!r} expected_iqr={want:.6g}")
            continue
        rel = abs(g - want) / max(abs(want), 1e-30)
        flag = "  <-- MISMATCH" if rel > 1e-9 else ""
        print(f"    {t}: got={g:.6g} expected_iqr={want:.6g} rel={rel:.3g}{flag}")
    return got


def drawn_halfrange(fig, ch_ix_in_picks):
    line = fig.mne.traces[ch_ix_in_picks]
    y = np.asarray(np.ma.getdata(line.get_ydata()), float)
    return float(np.ptp(y))


def scale_check(spec, label, **plot_kwargs):
    """Drawn peak-to-peak vs source ptp / (2 * scaling) * scale_factor."""
    case = build(spec)
    fig = open_browser(case, "matplotlib", n_epochs=len(case.lengths), **plot_kwargs)
    sc = dict(fig.mne.scalings)
    scale_factor = float(fig.mne.scale_factor)
    picks = np.asarray(fig.mne.picks)
    ch_types = np.asarray(fig.mne.ch_types)
    print(f"\n--- {label}: scalings={ {k: (f'{v:.4g}' if not isinstance(v, str) else v) for k, v in sc.items() if k in set(ch_types[picks])} } scale_factor={scale_factor}")
    for ii, p in enumerate(picks[:4]):
        t = ch_types[p]
        strip = np.concatenate([a[p] for a in case.source])
        # the browser removes DC over the loaded window, which does not change ptp
        want = np.ptp(strip) / (2 * sc[t]) * scale_factor
        got = drawn_halfrange(fig, ii)
        rel = abs(got - want) / max(abs(want), 1e-30)
        flag = "  <-- MISMATCH" if rel > 1e-6 else ""
        print(f"    trace {ii} ({t}): drawn ptp={got:.6g} expected={want:.6g} rel={rel:.3g}{flag}")
    close_fig(fig)


def subset_path():
    """Hit the ``preload is False`` epoch branch with ragged durations."""
    print("\n=== subset path (preload=False) ===")
    rng = np.random.default_rng(1)
    sfreq = 100.0
    n_ch = 4
    names = [f"EEG{i:03d}" for i in range(n_ch)]
    info = mne.create_info(names, sfreq, ["eeg"] * n_ch)
    raw = mne.io.RawArray(rng.standard_normal((n_ch, 40000)) * 1e-6, info, verbose=False)
    n_ep = 20
    onsets = np.arange(n_ep) * 1500 + 500
    events = np.column_stack([onsets, np.zeros(n_ep, int), np.ones(n_ep, int)])
    # 100:1 duration spread
    tmaxs = np.array([0.02 if i % 2 else 2.0 for i in range(n_ep)])
    try:
        ep = mne.Epochs(
            raw, events, tmin=np.zeros(n_ep), tmax=tmaxs, baseline=None,
            preload=False, verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    could not build non-preloaded ragged Epochs: {type(exc).__name__}: {exc}")
        return
    print(f"    preload={ep.preload} variable_duration={ep.variable_duration}")
    print(f"    per-epoch n_times: {[ep._n_times_per_epoch(i) for i in range(len(ep))]}")
    got = _compute_scalings("auto", ep.copy())
    # ground truth: full concatenated data
    full = ep.copy().load_data()
    strip = np.concatenate(list(full.get_data(copy=False)), axis=-1)
    want = float(np.diff(np.percentile(strip.ravel(), [25, 75]))[0])
    print(f"    auto eeg={got['eeg']:.6g}   full-data IQR={want:.6g}   ratio={got['eeg'] / want:.4f}")


def whitened(spec, label):
    print(f"\n=== whitening ({label}) ===")
    case = build(spec)
    ep = case.epochs
    n_ch = len(ep.ch_names)
    # a full-rank, single-channel-type covariance: ``_setup_plot_projector``
    # whitens per channel type, so a cross-type identity trips its own assert
    rng = np.random.default_rng(3)
    a = rng.standard_normal((n_ch, n_ch))
    cov = mne.Covariance(
        data=(a @ a.T) * 1e-13 + np.eye(n_ch) * 1e-13,
        names=list(ep.ch_names),
        bads=[],
        projs=[],
        nfree=1000,
        verbose=False,
    )
    try:
        fig = open_browser(
            case, "matplotlib", n_epochs=len(case.lengths), noise_cov=cov
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        print("    open with noise_cov FAILED:")
        traceback.print_exc(limit=6)
        return
    print(f"    whitened_ch_names={list(fig.mne.whitened_ch_names)[:5]}")
    print(f"    scalings['whitened']={fig.mne.scalings.get('whitened')}")
    y = np.asarray(np.ma.getdata(fig.mne.traces[0].get_ydata()), float)
    print(f"    trace 0 drawn ptp={np.ptp(y):.6g} finite={np.isfinite(y).all()}")
    try:
        fig._fake_keypress("w")
        y2 = np.asarray(np.ma.getdata(fig.mne.traces[0].get_ydata()), float)
        print(f"    after 'w': drawn ptp={np.ptp(y2):.6g} finite={np.isfinite(y2).all()}")
    except Exception:  # noqa: BLE001
        import traceback

        print("    'w' keypress FAILED:")
        traceback.print_exc(limit=6)
    close_fig(fig)


def main():
    assert_env()
    spread = Spec(lengths=(3, 300, 3, 300), name="a2_spread_100to1")
    spread_fixed = Spec(
        lengths=(300, 300, 300, 300), force_fixed=True, name="a2_spread_fixed"
    )
    mixed = Spec(
        lengths=(100, 250, 75), n_channels=16, mixed_types=True, name="a2_mixed"
    )
    mixed_fixed = Spec(
        lengths=(100, 100, 100), n_channels=16, mixed_types=True,
        force_fixed=True, name="a2_mixed_fixed",
    )

    print("=== auto scalings vs IQR of the source strip ===")
    report_auto(spread, "RAGGED 100:1 spread")
    report_auto(spread_fixed, "FIXED control")
    report_auto(mixed, "RAGGED mixed channel types")
    report_auto(mixed_fixed, "FIXED mixed channel types")

    print("\n=== drawn amplitude vs scalings ===")
    scale_check(spread, "RAGGED auto")
    scale_check(spread_fixed, "FIXED auto")
    scale_check(spread, "RAGGED dict", scalings=dict(eeg=20e-6))
    scale_check(mixed, "RAGGED mixed auto")
    scale_check(mixed, "RAGGED mixed dict", scalings=dict(eeg=20e-6, mag=4e-12, grad=4e-10, misc=1e-3))

    subset_path()
    whitened(Spec(lengths=(100, 250, 75, 180), n_channels=8), "RAGGED eeg")
    whitened(Spec(lengths=(100, 100, 100, 100), n_channels=8, force_fixed=True), "FIXED eeg")
    whitened(Spec(lengths=(3, 300, 3, 300), n_channels=8), "RAGGED 100:1 spread")


if __name__ == "__main__":
    main()
