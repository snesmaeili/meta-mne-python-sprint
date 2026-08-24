"""End-to-end walk through all three layers on real MNE data.

Uses `mne.datasets.sample`, so anyone can run it without our gait recordings.
Variable-duration epochs are created the way the architecture proposes: from
`Annotations` that carry a real `duration`, which `mne.Epochs` currently reads
the onset of and then discards.

Run:  python validation/end_to_end_sample_data.py
"""

from __future__ import annotations

import sys
import pathlib

import mne
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ragged_epochs import RaggedEpochs, RaggedTimesError  # noqa: E402
from ragged_epochs import _ops as ops  # noqa: E402
from ragged_epochs._align import landmark_warp, pad  # noqa: E402
from ragged_epochs._tfr import compute_tfr, warp_tfr  # noqa: E402

mne.set_log_level("ERROR")
RNG = np.random.default_rng(0)


def head(msg):
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}")


def _sample_path():
    """Locate MNE sample data, tolerating a stale MNE_DATA config entry.

    `mne.get_config('MNE_DATA')` can point at a drive that no longer exists;
    check the plausible roots for an already-downloaded copy before asking
    MNE to fetch one.
    """
    import os

    candidates = [
        os.environ.get("MNE_DATA"),
        mne.get_config("MNE_DATA"),
        "D:/mne_data",
        pathlib.Path.home() / "mne_data",
    ]
    for root in candidates:
        if not root:
            continue
        p = pathlib.Path(root) / "MNE-sample-data"
        if (p / "MEG" / "sample" / "sample_audvis_raw.fif").exists():
            return p
    return mne.datasets.sample.data_path()


def main():
    head("0. Real data, with genuinely variable trial durations")
    path = _sample_path()
    raw = mne.io.read_raw_fif(path / "MEG" / "sample" / "sample_audvis_raw.fif")
    events = mne.find_events(raw, stim_channel="STI 014", verbose=False)
    raw.pick(["eeg"]).load_data().filter(1.0, 40.0, verbose=False)

    events = events[np.isin(events[:, 2], [1, 2])][:40]
    onsets = events[:, 0] / raw.info["sfreq"] - raw.first_time

    # Variable "trial" durations, as in a self-paced or response-terminated
    # task: the process lasts as long as it lasts.
    durations = RNG.uniform(0.45, 0.95, len(onsets))
    # three internal landmarks per trial, at jittered fractions of the trial
    fracs = np.array([0.0, 0.25, 0.60, 1.0])
    landmarks = [
        np.sort(np.r_[0.0, d * (fracs[1:-1] + RNG.normal(0, 0.03, 2)), d])
        for d in durations
    ]

    annot = mne.Annotations(
        onset=onsets + raw.first_time, duration=durations,
        description=["trial"] * len(onsets),
    )
    raw.set_annotations(annot)
    print(f"raw: {len(raw.ch_names)} EEG channels @ {raw.info['sfreq']:.1f} Hz")
    print(f"annotations: {len(annot)} 'trial' events, "
          f"durations {durations.min():.3f}-{durations.max():.3f} s")

    head("1. LAYER 1 -- build from annotations, honouring duration")
    print("mne.Epochs docstring: 'the durations of the annotations are ignored "
          "in this case'")
    stock = mne.Epochs(raw, tmin=0, tmax=0.95, baseline=None, preload=True,
                       verbose=False)
    print(f"  mne.Epochs        -> {stock.get_data().shape}  "
          f"(one tmax for all; the variable part is gone)")

    ep = RaggedEpochs.from_annotations(raw, description="trial")
    print(f"  RaggedEpochs      -> {ep}")
    print(f"  durations         -> {np.round(ep.durations[:6], 3)} ... "
          f"({len(ep)} epochs)")
    print(f"  total samples kept: {ep.lengths.sum():,} vs "
          f"{len(ep) * ep.lengths.max():,} if padded to the longest "
          f"({100 * (1 - ep.lengths.sum() / (len(ep) * ep.lengths.max())):.1f}% "
          f"would be padding)")

    try:
        ep.times
    except RaggedTimesError as exc:
        print(f"\n  ep.times raises rather than guessing:\n    "
              + str(exc).splitlines()[0])

    head("2. LAYER 2 -- ragged-native operations, no kernel rewrites")
    ref = ops.set_eeg_reference(ep, "average")
    base = ops.apply_baseline(ref, baseline=(None, 0.1))
    print(f"  average reference + baseline -> {base}")

    for weighting in ("samples", "equal"):
        cov = ops.compute_covariance(base, weighting=weighting)
        print(f"  covariance (weighting={weighting:7s}) "
              f"trace={np.trace(cov):.3e}")
    print("  ^ the two differ: ragged data exposes the sample-vs-epoch")
    print("    weighting assumption that fixed-length epochs hide")

    psd, freqs, per_epoch = ops.compute_psd(base, fmin=2, fmax=40)
    print(f"  PSD on a common grid: {psd.shape}, "
          f"n_fft capped at the shortest epoch ({base.lengths.min()} samples)")

    head("3. LAYER 3 -- explicit alignment; nothing happens implicitly")
    padded, nave = pad(base)
    print(f"  pad()  -> uniform={padded.is_uniform}, "
          f"nave(t) from {nave.max()} down to {nave.min()}")
    print("           ^ agramfort's objection, handed over rather than hidden")

    # The shortest epoch bounds the frequency set -- ask for too much and the
    # policy check says so, with the number needed to fix it.
    try:
        compute_tfr(base, np.arange(6.0, 31.0, 1.0))
    except ValueError as exc:
        print("\n  compute_tfr refuses an impossible frequency set:")
        for line in str(exc).splitlines():
            print(f"    {line}")

    # take the check at its word: 11 Hz is the lowest that fits
    freqs = np.arange(11.0, 31.0, 1.0)
    tfr = compute_tfr(base, freqs, n_cycles=3.0)
    print(f"\n  compute_tfr(average=False) -> {tfr}")
    try:
        tfr.average()
    except RaggedTimesError as exc:
        print(f"  tfr.average() raises: {str(exc).splitlines()[0]}")

    aligned = warp_tfr(tfr, landmarks[: len(tfr)], target="median",
                       n_points=200, landmark_names=("on", "L1", "L2", "off"))
    print(f"\n  warp_tfr(target='median') -> {aligned}")
    print(f"  provenance: {aligned.alignment.summary()}")
    avg = aligned.average()
    print(f"  average() now defined -> {avg.shape} (n_channels, n_freqs, n_times)")

    head("4. Back to first-class MNE objects")
    etfr = aligned.to_mne()
    print(f"  {type(etfr).__name__}: data {etfr.data.shape}, "
          f"times {etfr.times[0]:.3f}-{etfr.times[-1]:.3f} s, "
          f"{len(etfr.freqs)} freqs")
    print(f"  isinstance of mne.time_frequency.EpochsTFR: "
          f"{isinstance(etfr, mne.time_frequency.EpochsTFR)}")

    head("5. The signal-vs-TFR distinction, on this data")
    sig_warped = landmark_warp(base, landmarks[: len(base)], target="median",
                               n_points=200)
    print(f"  signal-domain warp: warps_spectral_content="
          f"{sig_warped.alignment.warps_spectral_content}")
    print(f"  TFR-domain warp   : warps_spectral_content="
          f"{aligned.alignment.warps_spectral_content}")
    print("\n  Both produce a common axis. Only the TFR-domain one leaves the")
    print("  frequency axis meaningful -- see tests/test_frequency_preservation.py")

    print("\nEnd-to-end run complete.")


if __name__ == "__main__":
    main()
