"""V3 -- the parity oracle.

Every layer-2 operation must give the same answer as running stock MNE on one
epoch at a time. This is what makes "no kernel rewrites" a checkable claim
rather than a hope, and it is the argument that the ragged path does not
introduce a second, subtly different implementation of anything.

The comparison is deliberately against *per-epoch* MNE, not against MNE on
padded data -- padding changes filter edge behaviour and PSD normalisation, so
it would not be a fair oracle.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from ragged_epochs import RaggedEpochs
from ragged_epochs import _ops as ops
from ragged_epochs._tfr import compute_tfr

mne.set_log_level("ERROR")

SFREQ = 200.0
CH = ["C3", "Cz", "C4", "Pz"]
DURATIONS = (1.00, 1.37, 1.12, 0.83, 1.55)


@pytest.fixture
def epochs():
    info = mne.create_info(CH, SFREQ, "eeg")
    rng = np.random.default_rng(7)
    # Deliberately heteroscedastic: longer trials are not statistically
    # identical to shorter ones. That is the realistic case, and it is what
    # makes the sample- vs epoch-weighting choice have consequences.
    blocks = [
        rng.standard_normal((len(CH), int(round(d * SFREQ)))) * 1e-6 * (1.0 + 2.0 * k)
        for k, d in enumerate(DURATIONS)
    ]
    return RaggedEpochs(blocks, info)


def _per_epoch_mne(epochs, fn):
    """Run `fn` through a stock one-trial mne.EpochsArray, per epoch."""
    out = []
    for i in range(len(epochs)):
        d = epochs.get_data(i)[np.newaxis]
        ea = mne.EpochsArray(d, epochs.info, tmin=epochs.tmin[i], verbose=False)
        out.append(fn(ea)[0])
    return out


def test_filter_matches_mne(epochs):
    got = ops.filter(epochs, 1.0, 40.0)
    want = _per_epoch_mne(
        epochs, lambda ea: ea.copy().filter(1.0, 40.0, verbose=False).get_data()
    )
    for i, w in enumerate(want):
        np.testing.assert_allclose(got.get_data(i), w, rtol=1e-10, atol=1e-18)


def test_baseline_matches_mne(epochs):
    got = ops.apply_baseline(epochs, baseline=(None, 0.3))
    want = _per_epoch_mne(
        epochs,
        lambda ea: ea.copy().apply_baseline((None, 0.3), verbose=False).get_data(),
    )
    for i, w in enumerate(want):
        np.testing.assert_allclose(got.get_data(i), w, rtol=1e-10, atol=1e-18)


def test_average_reference_matches_mne(epochs):
    got = ops.set_eeg_reference(epochs, "average")
    want = _per_epoch_mne(
        epochs,
        lambda ea: ea.copy()
        .set_eeg_reference("average", projection=False, verbose=False)
        .get_data(),
    )
    for i, w in enumerate(want):
        np.testing.assert_allclose(got.get_data(i), w, rtol=1e-10, atol=1e-18)


def test_hilbert_envelope_matches_mne(epochs):
    got = ops.apply_hilbert(epochs, envelope=True)
    want = _per_epoch_mne(
        epochs,
        lambda ea: ea.copy().apply_hilbert(envelope=True, verbose=False).get_data(),
    )
    for i, w in enumerate(want):
        np.testing.assert_allclose(got.get_data(i), w, rtol=1e-10, atol=1e-18)


def test_tfr_matches_mne(epochs):
    freqs = np.arange(6.0, 31.0, 2.0)
    got = compute_tfr(epochs, freqs, n_cycles=freqs / 2.0)
    want = _per_epoch_mne(
        epochs,
        lambda ea: ea.compute_tfr(
            "morlet", freqs=freqs, n_cycles=freqs / 2.0,
            return_itc=False, average=False, verbose=False,
        ).get_data(),
    )
    for i, w in enumerate(want):
        np.testing.assert_allclose(got.get_data(i), w, rtol=1e-9, atol=0)


def test_uniform_ragged_epochs_equal_stock_epochs():
    """When durations happen to be equal, the ragged path is stock MNE.

    Issues #5794 and #11480 (per-trial time origin on rectangular data) are the
    degenerate case of this design, so the degenerate case had better be exact.
    """
    info = mne.create_info(CH, SFREQ, "eeg")
    rng = np.random.default_rng(11)
    data = rng.standard_normal((6, len(CH), 250)) * 1e-6
    ragged = RaggedEpochs(list(data), info, tmin=-0.2)
    stock = mne.EpochsArray(data, info, tmin=-0.2, verbose=False)

    np.testing.assert_allclose(ragged.times, stock.times, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        ragged.get_data(representation="dense"), stock.get_data(), rtol=0, atol=0
    )
    assert ragged.is_uniform


def test_concatenation_matches_mne_ica_input_layout(epochs):
    """`weighting='samples'` reproduces exactly what ICA.fit would see."""
    got, w = ops.concatenate_for_decomposition(epochs, weighting="samples")
    want = np.concatenate([epochs.get_data(i) for i in range(len(epochs))], axis=1)
    np.testing.assert_allclose(got, want, rtol=0, atol=0)
    np.testing.assert_allclose(w, np.ones(len(epochs)))


def test_equal_weighting_actually_equalises_trial_influence(epochs):
    """`weighting='equal'` removes the duration advantage of long trials."""
    _, w_samples = ops.concatenate_for_decomposition(epochs, weighting="samples")
    _, w_equal = ops.concatenate_for_decomposition(epochs, weighting="equal")

    lengths = epochs.lengths.astype(float)
    # under 'samples' a trial's influence is proportional to its length
    influence_samples = w_samples * lengths
    # under 'equal' every trial contributes the same total
    influence_equal = w_equal * lengths

    assert influence_samples.std() > 0
    np.testing.assert_allclose(
        influence_equal, influence_equal[0] * np.ones_like(influence_equal), rtol=1e-12
    )


def test_covariance_weighting_changes_the_answer(epochs):
    """The policy is not cosmetic -- it moves the numbers."""
    cov_s = ops.compute_covariance(epochs, weighting="samples")
    cov_e = ops.compute_covariance(epochs, weighting="equal")
    assert cov_s.shape == (len(CH), len(CH))
    np.testing.assert_allclose(cov_s, cov_s.T, rtol=1e-12)
    # atol must be 0 here: these are covariances of Volt-scale data (~1e-11),
    # so np.allclose's default atol=1e-8 would call any two of them equal.
    rel = np.abs(cov_s - cov_e).max() / np.abs(cov_s).max()
    assert rel > 0.01, (
        f"sample- vs epoch-weighted covariance differ by only {rel:.1%}; "
        "on ragged data with duration-correlated variance they should differ "
        "materially -- that difference is the assumption fixed-length epochs hide"
    )
