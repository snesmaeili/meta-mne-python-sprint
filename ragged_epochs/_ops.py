"""Layer 2 -- ragged-native operations.

Anything mathematically per-trial maps over epochs with no kernel rewrite.
mne-mobi's ``compute_single_ersp`` already proved this: it builds a one-trial
``Epochs`` at that cycle's own duration and runs an ordinary Morlet TFR on it.

The honest cost, which is easy to gloss over: this is a Python-level loop with
per-call overhead, and it is why ``compute_single_ersp`` is slow. So the
framing is deliberate -- **map-over-epochs is the correctness reference and the
parity oracle; vectorised fast paths get added where padding overhead is
acceptable, never the other way round.**

The ``*_policy`` functions are where the science lives. Fixed-length epochs
silently enforce ``sample weighting == epoch weighting``. Ragged trials break
that identity, so the choice has to be made explicit.
"""

from __future__ import annotations

import numpy as np

from ._container import RaggedEpochs

__all__ = [
    "map_epochs",
    "filter",
    "apply_baseline",
    "detrend",
    "apply_hilbert",
    "set_eeg_reference",
    "concatenate_for_decomposition",
    "compute_covariance",
    "compute_psd",
]


def map_epochs(epochs: RaggedEpochs, func, *, keep_length=True, **kwargs):
    """Apply ``func(data_2d, **kwargs)`` to each epoch independently.

    Parameters
    ----------
    func : callable
        Receives ``(n_channels, n_i)`` and returns an array with the same
        number of channels.
    keep_length : bool
        Assert that the output length matches the input. Set False for
        operations that legitimately change it (decimation, cropping).
    """
    blocks = []
    for i in range(len(epochs)):
        d = epochs.get_data(i)
        out = np.asarray(func(d, **kwargs))
        if out.shape[0] != d.shape[0]:
            raise ValueError(
                f"func changed the channel count on epoch {i}: "
                f"{d.shape[0]} -> {out.shape[0]}"
            )
        if keep_length and out.shape[-1] != d.shape[-1]:
            raise ValueError(
                f"func changed the length of epoch {i}: "
                f"{d.shape[-1]} -> {out.shape[-1]}. Pass keep_length=False "
                "if that is intended."
            )
        blocks.append(out)
    return RaggedEpochs(
        blocks, epochs.info, epochs.tmin,
        events=epochs.events, event_id=epochs.event_id,
        metadata=epochs.metadata, alignment=epochs.alignment,
    )


# ------------------------------------------------------- naturally ragged
def filter(epochs, l_freq, h_freq, **kwargs):
    """Band-pass each epoch independently.

    Note the edge effects are per-epoch and therefore duration-dependent: a
    0.8 s cycle and a 2.0 s cycle do not get the same effective filter. That is
    true of fixed-length epoching too, it is just usually invisible. Filtering
    the continuous ``Raw`` before epoching remains the better practice.
    """
    import mne

    kwargs.setdefault("verbose", False)
    # Epochs.filter defaults to pad="edge"; filter_data defaults to
    # "reflect_limited". Match the Epochs behaviour so the parity oracle in
    # tests/test_parity_with_mne.py is a like-for-like comparison.
    kwargs.setdefault("pad", "edge")
    return map_epochs(
        epochs,
        lambda d: mne.filter.filter_data(d, epochs.sfreq, l_freq, h_freq, **kwargs),
    )


def apply_baseline(epochs, baseline=(None, 0.0), mode="mean"):
    """Per-epoch baseline correction over a window in seconds."""

    def _one(d, times):
        lo = times[0] if baseline[0] is None else baseline[0]
        hi = times[-1] if baseline[1] is None else baseline[1]
        m = (times >= lo) & (times <= hi)
        if not m.any():
            raise ValueError(f"Baseline window {baseline} is empty for an epoch.")
        base = d[:, m].mean(axis=1, keepdims=True)
        if mode == "mean":
            return d - base
        if mode == "ratio":
            return d / base
        raise ValueError(f"Unknown mode {mode!r}.")

    blocks = [_one(epochs.get_data(i), epochs.get_times(i)) for i in range(len(epochs))]
    return RaggedEpochs(
        blocks, epochs.info, epochs.tmin,
        events=epochs.events, event_id=epochs.event_id,
        metadata=epochs.metadata, alignment=epochs.alignment,
    )


def detrend(epochs, order=1):
    """Remove a per-epoch polynomial trend."""

    def _one(d):
        n = d.shape[-1]
        x = np.linspace(-1, 1, n)
        coef = np.polynomial.polynomial.polyfit(x, d.T, order)
        return d - np.polynomial.polynomial.polyval(x, coef)

    return map_epochs(epochs, _one)


def apply_hilbert(epochs, envelope=False, n_fft="auto"):
    """Analytic signal (or its envelope) per epoch.

    Matches ``Epochs.apply_hilbert``: zero-pad to ``n_fft`` (``'auto'`` picks
    the next fast FFT length), transform, then cut back. The padding length is
    duration-dependent, which is a good example of a per-trial operation whose
    numerics legitimately differ between a 0.8 s and a 2.0 s epoch.
    """
    # MNE defines its own next_fast_len (2/3/5-smooth only), which differs
    # from scipy.fft.next_fast_len (also allows 7 and 11). Use MNE's so the
    # padding length matches Epochs.apply_hilbert exactly.
    from mne.filter import next_fast_len
    from scipy.signal import hilbert

    def _one(d):
        n = d.shape[-1]
        if n_fft == "auto":
            n_pad = next_fast_len(n)
        elif n_fft is None:
            n_pad = n
        else:
            n_pad = int(n_fft)
        h = hilbert(d, N=n_pad, axis=-1)[..., :n]
        return np.abs(h) if envelope else h

    return map_epochs(epochs, _one)


def set_eeg_reference(epochs, ref_channels="average"):
    """Re-reference. Purely spatial, so duration is irrelevant."""
    if ref_channels == "average":
        return map_epochs(epochs, lambda d: d - d.mean(axis=0, keepdims=True))
    names = epochs.ch_names
    idx = [names.index(c) for c in ref_channels]
    return map_epochs(epochs, lambda d: d - d[idx].mean(axis=0, keepdims=True))


# ------------------------------------------------- ragged WITH a policy
def _weights(epochs, weighting):
    """Per-epoch multiplicative weights implementing the chosen policy."""
    lengths = epochs.lengths.astype(float)
    if weighting == "samples":
        return np.ones(len(epochs))
    if weighting == "equal":
        # down-weight long epochs so every trial contributes equally
        return lengths.mean() / lengths
    raise ValueError(
        f"weighting must be 'samples' or 'equal', got {weighting!r}."
    )


def concatenate_for_decomposition(epochs, weighting="samples"):
    """Concatenate epochs into ``(n_channels, sum(lengths))`` for ICA.

    ``mne.preprocessing.ICA.fit`` already reshapes ``Epochs`` to
    ``(n_channels, n_epochs * n_times)``, so nothing about ICA mathematically
    requires equal lengths -- ragged ICA is natural.

    What ragged data *does* force into the open is a question fixed-length
    epochs hide: should a 5 s trial contribute five times the influence of a
    1 s trial?

    ``weighting='samples'``
        Every sample counts once. Long trials dominate. This is the implicit
        behaviour you get from naive concatenation, and it is the right answer
        when the goal is to model the data-generating process.
    ``weighting='equal'``
        Each trial contributes equally regardless of duration. Right when
        trials are experimental units and duration is a nuisance variable.

    There is no defensible default, which is why this returns the weights
    alongside the data instead of quietly picking one.
    """
    w = _weights(epochs, weighting)
    blocks = [epochs.get_data(i) * np.sqrt(w[i]) for i in range(len(epochs))]
    return np.concatenate(blocks, axis=1), w


def compute_covariance(epochs, weighting="samples"):
    """Sensor covariance across ragged epochs, with the weighting stated.

    Same question as ICA, and it propagates further: the covariance scales the
    noise model used by the inverse operator.
    """
    w = _weights(epochs, weighting)
    n_ch = epochs.get_data(0).shape[0]
    cov = np.zeros((n_ch, n_ch))
    total = 0.0
    for i in range(len(epochs)):
        d = epochs.get_data(i)
        d = d - d.mean(axis=1, keepdims=True)
        cov += w[i] * (d @ d.T)
        total += w[i] * d.shape[1]
    return cov / total


def compute_psd(epochs, fmin=0.0, fmax=np.inf, n_fft=None, weighting="samples"):
    """Welch PSD per epoch on a common frequency grid.

    The policy here is the frequency grid. Epochs of different duration have
    different natural frequency resolution; putting them on one grid means
    fixing ``n_fft`` for all of them, which caps resolution at the shortest
    epoch. Stating that beats letting each epoch return a different ``freqs``
    and silently interpolating later.
    """
    from mne.time_frequency import psd_array_welch

    if n_fft is None:
        n_fft = int(epochs.lengths.min())
    w = _weights(epochs, weighting)

    psds, freqs = [], None
    for i in range(len(epochs)):
        p, f = psd_array_welch(
            epochs.get_data(i), epochs.sfreq, fmin=fmin, fmax=fmax,
            n_fft=n_fft, n_per_seg=n_fft, verbose=False,
        )
        psds.append(p)
        freqs = f
    stacked = np.stack(psds)  # (n_epochs, n_channels, n_freqs)
    avg = np.average(stacked, axis=0, weights=w)
    return avg, freqs, stacked
