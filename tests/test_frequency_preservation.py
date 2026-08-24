"""V1 -- the frequency-preservation test.

The single most important test in the prototype. It encodes, as an executable
invariant, why signal-domain and TF-domain warping are not interchangeable.

Two trials carry the *same* 10 Hz oscillation and differ only in duration.
Normalising them to a common length can be done two ways:

    path A   warp the signal, then transform     -> apparent frequency shifts
    path B   transform, then warp the TF axis    -> frequency axis untouched

Path A is what mne-mobi's ``channel_tfr_general.warp_sig`` does. Path B is what
EEGLAB's ``newtimef`` does and what the gait-EEG literature means by
time-warped ERSP. If the quantity of interest is spectral power, path A
reports the wrong frequency.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from ragged_epochs import RaggedEpochs
from ragged_epochs._align import piecewise_linear_warp
from ragged_epochs._tfr import compute_tfr, warp_tfr

mne.set_log_level("ERROR")

SFREQ = 500.0
F_OSC = 10.0
FREQS = np.arange(4.0, 21.0, 0.5)
DURATIONS = (1.0, 2.0)  # a 2x duration ratio makes the effect unmissable


def _oscillation(duration, sfreq=SFREQ, freq=F_OSC, seed=0):
    """A pure `freq` Hz oscillation lasting `duration` seconds."""
    n = int(round(duration * sfreq))
    t = np.arange(n) / sfreq
    rng = np.random.default_rng(seed)
    return (np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(n))[None, :]


def _peak_frequency(power, freqs):
    """Frequency of maximum power, averaged over the middle half of the trial."""
    n = power.shape[-1]
    core = power[..., n // 4 : 3 * n // 4].mean(axis=-1)
    return float(freqs[np.argmax(core.ravel())])


@pytest.fixture
def epochs():
    info = mne.create_info(["Cz"], SFREQ, "eeg")
    blocks = [_oscillation(d, seed=i) for i, d in enumerate(DURATIONS)]
    return RaggedEpochs(blocks, info)


def test_native_epochs_both_show_10hz(epochs):
    """Sanity: before any warping, both trials peak at the true frequency."""
    tfr = compute_tfr(epochs, FREQS)
    for i in range(len(epochs)):
        assert _peak_frequency(tfr.get_data(i), FREQS) == pytest.approx(F_OSC, abs=0.5)


def test_path_A_signal_warp_shifts_apparent_frequency(epochs):
    """Warping the signal before the transform rescales apparent frequency.

    The 2 s trial is compressed onto the 1 s grid, so its 10 Hz oscillation
    completes twice as many cycles per unit of the new time axis and reads as
    ~20 Hz. This is a real distortion, not a numerical artefact.
    """
    target = np.array([0.0, DURATIONS[0]])  # normalise everything to 1 s
    n_out = int(DURATIONS[0] * SFREQ)

    warped = [
        piecewise_linear_warp(
            epochs.get_data(i), np.array([0.0, epochs.durations[i]]),
            target, n_out, SFREQ,
        )
        for i in range(len(epochs))
    ]
    info = mne.create_info(["Cz"], SFREQ, "eeg")
    tfr = compute_tfr(RaggedEpochs(warped, info), FREQS)

    peak_short = _peak_frequency(tfr.get_data(0), FREQS)
    peak_long = _peak_frequency(tfr.get_data(1), FREQS)

    assert peak_short == pytest.approx(F_OSC, abs=0.5)
    # the stretched trial is misreported by roughly the duration ratio
    ratio = DURATIONS[1] / DURATIONS[0]
    assert peak_long == pytest.approx(F_OSC * ratio, rel=0.15)
    assert abs(peak_long - peak_short) > 5.0, (
        "signal-domain warping should visibly move the spectral peak"
    )


def test_path_B_tfr_warp_preserves_frequency(epochs):
    """Warping the TF representation moves energy in time, not in frequency."""
    tfr = compute_tfr(epochs, FREQS)
    landmarks = [np.array([0.0, d]) for d in epochs.durations]
    aligned = warp_tfr(tfr, landmarks, target="median", n_points=500)

    peaks = [_peak_frequency(aligned.get_data(i), FREQS) for i in range(len(aligned))]
    for p in peaks:
        assert p == pytest.approx(F_OSC, abs=0.5)
    assert peaks[0] == peaks[1], "both trials must report the same frequency"


def test_tfr_warp_yields_a_common_axis_so_average_is_defined(epochs):
    """The point of aligning: only then does averaging across trials mean anything."""
    from ragged_epochs import RaggedTimesError

    tfr = compute_tfr(epochs, FREQS)
    with pytest.raises(RaggedTimesError, match="Cannot average"):
        tfr.average()

    aligned = warp_tfr(tfr, [np.array([0.0, d]) for d in epochs.durations],
                       n_points=500)
    avg = aligned.average()
    assert avg.shape == (1, len(FREQS), 500)
    assert _peak_frequency(avg, FREQS) == pytest.approx(F_OSC, abs=0.5)


def test_provenance_flags_the_distorting_path(epochs):
    """The object says which path produced it, so a reader can tell."""
    from ragged_epochs._align import landmark_warp

    signal_warped = landmark_warp(
        epochs, [np.array([0.0, d]) for d in epochs.durations], n_points=500
    )
    assert signal_warped.alignment.warps_spectral_content is True
    assert "shifts apparent frequency" in signal_warped.alignment.summary()

    tfr_warped = warp_tfr(
        compute_tfr(epochs, FREQS),
        [np.array([0.0, d]) for d in epochs.durations], n_points=500,
    )
    assert tfr_warped.alignment.warps_spectral_content is False


def test_complex_tfr_warp_preserves_phase_continuity(epochs):
    """Phase must be warped circularly, not linearly.

    Linear interpolation between +179 deg and -179 deg gives 0 deg instead of
    180 deg. EEGLAB uses ``angtimewarp`` for exactly this reason.
    """
    tfr = compute_tfr(epochs, FREQS, output="complex")
    aligned = warp_tfr(tfr, [np.array([0.0, d]) for d in epochs.durations],
                       n_points=500)
    z = aligned.get_data(0)
    f_idx = int(np.argmin(np.abs(FREQS - F_OSC)))
    phase = np.angle(z[0, f_idx])
    # a clean oscillation advances phase monotonically; wrapped jumps aside,
    # the unwrapped derivative must not change sign
    d = np.diff(np.unwrap(phase))
    assert np.median(d) > 0
    assert (d > 0).mean() > 0.95, "circular warp should preserve phase advance"
