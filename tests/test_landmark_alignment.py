"""V2 -- landmark alignment round-trip.

Synthetic gait cycles with the canonical five anchors::

    RHS ---- LTO -------- LHS ---- RTO -------- RHS
     0%      ~12%          50%      ~62%        100%

Every cycle has a different duration and jittered internal timing. After
alignment the landmarks must coincide, the original durations must still be
recoverable, and the provenance must describe what happened.

This file also pins C2: the warp target defaults to the **median** landmark
latencies, as EEGLAB does. mne-mobi's ``warp_sig`` used evenly-spaced
positions, which forces the five gait anchors to 25/25/25/25 and destroys the
real stance/swing split.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from ragged_epochs import RaggedEpochs
from ragged_epochs._align import (
    common_crop,
    duration_normalise,
    landmark_warp,
    pad,
    piecewise_linear_warp,
    resolve_target_landmarks,
)

mne.set_log_level("ERROR")

SFREQ = 500.0
CH = ["C3", "Cz", "C4"]
LANDMARKS = ("RHS", "LTO", "LHS", "RTO", "RHS_next")
#: canonical gait fractions -- stance is ~60% of the cycle, swing ~40%
FRACTIONS = np.array([0.00, 0.12, 0.50, 0.62, 1.00])
N_CYCLES = 24


@pytest.fixture
def gait():
    """24 gait cycles, durations 0.85-1.35 s, jittered internal landmarks."""
    rng = np.random.default_rng(42)
    info = mne.create_info(CH, SFREQ, "eeg")
    durations = rng.uniform(0.85, 1.35, N_CYCLES)

    blocks, landmarks = [], []
    for d in durations:
        jitter = np.r_[0.0, rng.normal(0, 0.015, 3), 0.0]
        lm = d * (FRACTIONS + jitter)
        lm = np.sort(lm)
        lm[0], lm[-1] = 0.0, d
        n = int(round(d * SFREQ))
        t = np.arange(n) / SFREQ
        # a smooth waveform whose features sit ON the landmarks, so alignment
        # is checkable from the data and not only from the bookkeeping
        sig = sum(np.exp(-0.5 * ((t - m) / 0.02) ** 2) for m in lm[1:-1])
        blocks.append(np.tile(sig, (len(CH), 1)) + 0.01 * rng.standard_normal((len(CH), n)))
        landmarks.append(lm)
    return RaggedEpochs(blocks, info), landmarks


def test_input_really_is_ragged(gait):
    epochs, _ = gait
    assert len(np.unique(epochs.lengths)) > 1
    assert epochs.durations.min() < epochs.durations.max()
    assert not epochs.is_uniform


def test_landmarks_coincide_after_warping(gait):
    """The bookkeeping check: every trial's landmarks map onto the target."""
    epochs, landmarks = gait
    dst = resolve_target_landmarks(landmarks, "median")
    n_out = 1000
    u = np.linspace(dst[0], dst[-1], n_out)

    worst_ms = 0.0
    for i in range(len(epochs)):
        t = np.arange(epochs.lengths[i]) / SFREQ
        # probe encodes "which landmark index am I at"
        probe = np.interp(t, landmarks[i], np.arange(len(LANDMARKS), dtype=float))
        out = piecewise_linear_warp(probe[None], landmarks[i], dst, n_out, SFREQ)[0]
        recovered = np.interp(dst, u, out)
        err_units = np.abs(recovered - np.arange(len(LANDMARKS)))
        # convert landmark-index error to milliseconds via the local slope
        slopes = np.diff(np.arange(len(LANDMARKS))) / np.diff(dst)
        worst_ms = max(worst_ms, (err_units[1:-1] / slopes[:-1]).max() * 1000)

    # discretisation-limited: the error is one source sample, not a bias
    one_sample_ms = 1000.0 / SFREQ
    assert worst_ms < 1.5 * one_sample_ms, f"{worst_ms:.2f} ms > 1.5 samples"


def test_landmark_error_is_discretisation_not_bias():
    """Halving the sample period must halve the landmark error."""
    rng = np.random.default_rng(3)
    durations = rng.uniform(0.9, 1.3, 8)
    landmarks = [d * FRACTIONS for d in durations]
    dst = resolve_target_landmarks(landmarks, "median")

    errs = []
    for sfreq in (250.0, 1000.0):
        worst = 0.0
        for i, d in enumerate(durations):
            n = int(round(d * sfreq))
            t = np.arange(n) / sfreq
            probe = np.interp(t, landmarks[i], np.arange(5.0))
            out = piecewise_linear_warp(probe[None], landmarks[i], dst, 4000, sfreq)[0]
            u = np.linspace(dst[0], dst[-1], 4000)
            worst = max(worst, np.abs(np.interp(dst, u, out) - np.arange(5.0)).max())
        errs.append(worst)
    assert errs[1] < errs[0] / 3.0, f"error did not scale with sfreq: {errs}"


def test_waveform_features_align(gait):
    """The data check: the bumps line up after warping, and did not before."""
    epochs, landmarks = gait
    aligned = landmark_warp(epochs, landmarks, target="median", n_points=600,
                            landmark_names=LANDMARKS)

    stack = np.stack([aligned.get_data(i)[0] for i in range(len(aligned))])
    # across-trial variance at each time point, normalised by total variance
    aligned_ratio = stack.var(axis=0).mean() / stack.var()

    n_min = int(epochs.lengths.min())
    raw_stack = np.stack([epochs.get_data(i)[0, :n_min] for i in range(len(epochs))])
    raw_ratio = raw_stack.var(axis=0).mean() / raw_stack.var()

    assert aligned_ratio < raw_ratio, (
        f"alignment should reduce across-trial dispersion "
        f"({aligned_ratio:.3f} vs {raw_ratio:.3f})"
    )


def test_durations_survive_alignment(gait):
    """After warping, the experimental durations are still recoverable."""
    epochs, landmarks = gait
    before = epochs.durations.copy()
    aligned = landmark_warp(epochs, landmarks, n_points=600)

    assert aligned.is_uniform  # it now has a common axis
    np.testing.assert_allclose(aligned.alignment.original_duration, before)
    assert aligned.alignment.original_duration.min() < aligned.alignment.original_duration.max()


def test_provenance_is_complete(gait):
    epochs, landmarks = gait
    aligned = landmark_warp(epochs, landmarks, target="median", n_points=600,
                            landmark_names=LANDMARKS)
    rec = aligned.alignment
    assert rec.method == "piecewise-linear"
    assert rec.domain == "signal"
    assert rec.target_rule == "median"
    assert rec.landmark_names == LANDMARKS
    assert len(rec.original_landmarks) == len(epochs)
    np.testing.assert_allclose(
        rec.target_landmarks, resolve_target_landmarks(landmarks, "median")
    )
    assert rec.warps_spectral_content is True  # signal-domain warp


def test_no_fake_sfreq(gait):
    """The percent axis is never smuggled in as a sampling frequency.

    mne-mobi's ``time_warp_epochs`` set ``info["sfreq"] = n_points - 1`` to fake
    a 0-100% axis. After that, "600 ms" and "60% of trial" were the same number
    in the same field. Here ``sfreq`` stays a real rate over a real span.
    """
    epochs, landmarks = gait
    n_points = 600
    aligned = landmark_warp(epochs, landmarks, n_points=n_points)
    dst = aligned.alignment.target_landmarks

    assert aligned.sfreq != n_points - 1
    expected = (n_points - 1) / (dst[-1] - dst[0])
    assert aligned.sfreq == pytest.approx(expected)
    # and the resulting time axis really spans the target landmark range
    assert aligned.times[0] == pytest.approx(dst[0])
    assert aligned.times[-1] == pytest.approx(dst[-1])


# ------------------------------------------------------------------- C2
def test_median_target_preserves_stance_swing_proportions(gait):
    """C2: 'median' keeps the real gait proportions; 'uniform' destroys them."""
    _, landmarks = gait
    med = resolve_target_landmarks(landmarks, "median")
    uni = resolve_target_landmarks(landmarks, "uniform")

    med_frac = (med - med[0]) / (med[-1] - med[0])
    uni_frac = (uni - uni[0]) / (uni[-1] - uni[0])

    np.testing.assert_allclose(med_frac, FRACTIONS, atol=0.03)
    np.testing.assert_allclose(uni_frac, np.linspace(0, 1, 5), atol=1e-12)

    # the toe-off anchor lands at ~12% under median and 25% under uniform
    assert med_frac[1] == pytest.approx(0.12, abs=0.03)
    assert uni_frac[1] == pytest.approx(0.25, abs=1e-9)


def test_uniform_target_must_be_asked_for_by_name(gait):
    """The distorting behaviour is reachable, but never the default."""
    _, landmarks = gait
    default = resolve_target_landmarks(landmarks)
    median = resolve_target_landmarks(landmarks, "median")
    np.testing.assert_allclose(default, median)


def test_mismatched_landmark_counts_are_refused():
    """Cycles with 4 and 5 anchors must not be warped onto a common target.

    mne-mobi's ``channel_tfr_general`` accepted 2-5 anchors per cycle and mapped
    whatever it found onto evenly-spaced positions, so a cycle missing LHS had
    its RTO silently aligned to another cycle's LHS. ``keep_only_five_anchor``
    existed as a workaround; refusing is better than a flag.
    """
    landmarks = [np.array([0.0, 0.2, 0.6, 1.0]), np.array([0.0, 0.1, 0.5, 0.7, 1.1])]
    with pytest.raises(ValueError, match="same number of landmarks"):
        resolve_target_landmarks(landmarks)


def test_non_monotonic_landmarks_are_refused():
    with pytest.raises(ValueError, match="strictly increasing"):
        piecewise_linear_warp(
            np.zeros((1, 100)), [0.0, 0.5, 0.4, 1.0], [0.0, 0.3, 0.6, 1.0], 50, 100.0
        )


# ------------------------------------------------- other strategies
def test_common_crop_is_lossy_and_uniform(gait):
    epochs, _ = gait
    cropped = common_crop(epochs)
    assert cropped.is_uniform
    assert cropped.lengths[0] == epochs.lengths.min()
    assert cropped.alignment.method == "common-crop"
    np.testing.assert_allclose(cropped.alignment.original_duration, epochs.durations)


def test_pad_returns_time_dependent_nave(gait):
    """O1: under padding, nave is a function of time. Hand it over, don't hide it."""
    epochs, _ = gait
    padded, nave = pad(epochs)

    assert padded.is_uniform
    assert nave.shape == (int(epochs.lengths.max()),)
    assert nave[0] == len(epochs)              # every trial covers t=0
    assert nave[-1] == 1                       # only the longest reaches the end
    assert nave.min() < nave.max(), "this is precisely agramfort's objection"
    # monotonically non-increasing: trials drop out as time goes on
    assert np.all(np.diff(nave) <= 0)


def test_duration_normalise_is_the_two_landmark_case(gait):
    epochs, _ = gait
    a = duration_normalise(epochs, n_points=400)
    b = landmark_warp(
        epochs, [np.array([0.0, d]) for d in epochs.durations], n_points=400
    )
    for i in range(len(epochs)):
        np.testing.assert_allclose(a.get_data(i), b.get_data(i), rtol=1e-12)
    assert a.alignment.method == "duration-normalise"


def test_landmarks_can_come_from_metadata(gait):
    """Reuse: this is the shape Epochs.add_annotations_to_metadata() produces."""
    import pandas as pd

    epochs, landmarks = gait
    epochs.metadata = pd.DataFrame({"annot_onset": [list(lm) for lm in landmarks]})
    from_md = landmark_warp(epochs, "annot_onset", n_points=400)
    direct = landmark_warp(epochs, landmarks, n_points=400)
    for i in range(len(epochs)):
        np.testing.assert_allclose(from_md.get_data(i), direct.get_data(i))
