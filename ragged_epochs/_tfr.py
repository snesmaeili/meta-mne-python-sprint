"""Ragged time-frequency representation, and TFR-domain alignment.

This is where the architecture earns its keep. The frequency axis is common
across epochs; only time is ragged. That is perfectly coherent, and it is the
representation the gait-EEG literature has been reaching for since Gwin et al.
(2011):

    epoch 0:  (n_channels, n_freqs, 417)
    epoch 1:  (n_channels, n_freqs, 463)
    epoch 2:  (n_channels, n_freqs, 401)

``average()`` over that is meaningless -- time index 350 is not the same phase
of the task in every trial -- so it raises and points at ``warp_tfr``.
"""

from __future__ import annotations

import numpy as np

from ._align import _warp_complex, piecewise_linear_warp, resolve_target_landmarks
from ._provenance import AlignmentRecord

__all__ = ["RaggedEpochsTFR", "compute_tfr", "warp_tfr"]


class RaggedEpochsTFR:
    """Per-epoch ``(n_channels, n_freqs, n_i)`` time-frequency data."""

    def __init__(self, data, info, freqs, tmin, *, output="power",
                 alignment=None, events=None, metadata=None, sfreq=None):
        self._data = list(data)
        self.info = info
        self.freqs = np.asarray(freqs, dtype=float)
        self._tmin = np.broadcast_to(
            np.asarray(tmin, dtype=float), (len(self._data),)
        ).copy()
        self.output = output
        self.alignment = alignment
        self.events = events
        self.metadata = metadata
        self._sfreq = float(sfreq if sfreq is not None else info["sfreq"])

    def __len__(self):
        return len(self._data)

    @property
    def lengths(self):
        return np.array([d.shape[-1] for d in self._data], dtype=np.int64)

    @property
    def sfreq(self):
        return self._sfreq

    @property
    def is_uniform(self):
        return bool(
            len(np.unique(self.lengths)) == 1
            and np.allclose(self._tmin, self._tmin[0])
        )

    @property
    def times(self):
        from ._container import RaggedTimesError

        if self.is_uniform:
            return self._tmin[0] + np.arange(self.lengths[0]) / self._sfreq
        raise RaggedTimesError(
            f"These {len(self)} TFRs do not share a time axis. Use "
            ".get_times(i), or warp_tfr(...) to put them on a common "
            "landmark-referenced axis first."
        )

    def get_times(self, epoch=None):
        if epoch is None:
            return [self.get_times(i) for i in range(len(self))]
        return self._tmin[epoch] + np.arange(self.lengths[epoch]) / self._sfreq

    def get_data(self, epoch=None):
        return self._data[epoch] if epoch is not None else list(self._data)

    def apply_baseline(self, mode="logratio", baseline=None):
        """Per-epoch baseline. Naturally ragged: each trial is its own window.

        With ``baseline=None`` the whole trial is the baseline -- the
        single-trial normalisation of Grandchamp & Delorme, and what mne-mobi's
        ``compute_single_ersp`` does when it subtracts the per-cycle mean.
        """
        if self.output != "power":
            raise ValueError("apply_baseline needs output='power'.")
        out = []
        for i, d in enumerate(self._data):
            if baseline is None:
                base = d.mean(axis=-1, keepdims=True)
            else:
                t = self.get_times(i)
                m = (t >= baseline[0]) & (t <= baseline[1])
                if not m.any():
                    raise ValueError(f"Baseline {baseline} empty for epoch {i}.")
                base = d[..., m].mean(axis=-1, keepdims=True)
            if mode == "logratio":
                out.append(10 * np.log10(d / base))
            elif mode == "ratio":
                out.append(d / base)
            elif mode == "mean":
                out.append(d - base)
            else:
                raise ValueError(f"Unknown mode {mode!r}.")
        return RaggedEpochsTFR(
            out, self.info, self.freqs, self._tmin, output="power",
            alignment=self.alignment, events=self.events,
            metadata=self.metadata, sfreq=self._sfreq,
        )

    def average(self):
        """Average across epochs. Requires a common time axis.

        Refusing this on ragged input is the whole point. Silently averaging
        misaligned processes is exactly what issue #5612 says NaN-padding fails
        to fix, and what makes a time-dependent ``nave`` sneak into results.
        """
        from ._container import RaggedTimesError

        if not self.is_uniform:
            raise RaggedTimesError(
                "Cannot average TFRs of unequal duration: time index k is not "
                "the same phase of the task in every trial. Align first:\n"
                "  warp_tfr(tfr, landmarks, target='median')\n"
                "then average. See tests/test_frequency_preservation.py for "
                "why this must happen in the TF domain, not on the signal."
            )
        return np.mean(np.stack(self._data), axis=0)

    def to_mne(self):
        """Convert an aligned TFR to a first-class ``mne.time_frequency`` object."""
        from mne.time_frequency import EpochsTFRArray

        if not self.is_uniform:
            raise ValueError("Only an aligned (uniform) TFR converts to EpochsTFR.")
        info = self.info.copy()
        with info._unlock():
            info["sfreq"] = self._sfreq
        return EpochsTFRArray(
            info=info,
            data=np.stack(self._data),
            times=self.times,
            freqs=self.freqs,
            method="morlet",
        )

    def __repr__(self):
        lo, hi = self.lengths.min(), self.lengths.max()
        span = f"{lo}" if lo == hi else f"{lo}-{hi}"
        a = f", aligned: {self.alignment.method}" if self.alignment else ""
        return (
            f"<RaggedEpochsTFR | {len(self)} epochs, {len(self.freqs)} freqs "
            f"({self.freqs[0]:g}-{self.freqs[-1]:g} Hz), {span} samples, "
            f"output={self.output}{a}>"
        )


def _wavelet_length(freq, n_cycles, sfreq):
    """Morlet wavelet length in samples, matching MNE's construction."""
    sigma_t = n_cycles / (2.0 * np.pi * freq)
    return 2 * int(sigma_t * 5.0 * sfreq) + 1


def _check_wavelets_fit(freqs, n_cycles, sfreq, n_min):
    """Refuse frequencies whose wavelet is longer than the shortest epoch."""
    n_cycles = np.broadcast_to(np.asarray(n_cycles, dtype=float), freqs.shape)
    lengths = np.array(
        [_wavelet_length(f, c, sfreq) for f, c in zip(freqs, n_cycles)]
    )
    bad = lengths > n_min
    if not bad.any():
        return

    ok = freqs[~bad]
    hint = (
        f"lowest usable frequency here is {ok.min():g} Hz"
        if ok.size
        else "no requested frequency fits"
    )
    # with n_cycles proportional to freq, wavelet length is frequency-
    # independent, so raising fmin does not help -- say so explicitly
    constant = len(np.unique(lengths)) == 1
    remedy = (
        "n_cycles is proportional to freq here, so every wavelet has the same "
        "length and raising fmin will not help: lower n_cycles, or drop the "
        "shortest epochs."
        if constant
        else f"Raise fmin ({hint}), lower n_cycles, or drop the shortest epochs."
    )
    raise ValueError(
        f"{bad.sum()} of {len(freqs)} requested frequencies need a wavelet "
        f"longer than the shortest epoch ({lengths.max()} > {n_min} samples).\n"
        f"The shortest epoch bounds the whole set, because every epoch must "
        f"yield the same frequency axis.\n{remedy}"
    )


def compute_tfr(epochs, freqs, *, n_cycles=None, output="power", zero_mean=True):
    """Morlet TFR of every epoch at its own duration.

    Layer 2 in action: no kernel is rewritten. ``mne.time_frequency.
    tfr_array_morlet`` is called once per epoch on a perfectly ordinary
    ``(n_channels, n_times)`` array. mne-mobi's ``compute_single_ersp`` already
    proved this works; the honest cost is a Python-level loop, which is why
    this is the correctness reference rather than the fast path.
    """
    from mne.time_frequency import tfr_array_morlet

    freqs = np.asarray(freqs, dtype=float)
    if n_cycles is None:
        n_cycles = freqs / 2.0
    sfreq = epochs.sfreq

    # Policy check: the SHORTEST epoch bounds which frequencies exist at all.
    # This is a constraint ragged data makes visible; fixed-length epoching has
    # it too, it is just decided once at tmin/tmax time. Fail with the number
    # the user needs rather than letting MNE report it per-epoch.
    _check_wavelets_fit(freqs, n_cycles, sfreq, int(epochs.lengths.min()))

    out = []
    for i in range(len(epochs)):
        d = epochs.get_data(i)[np.newaxis]  # (1, n_channels, n_times)
        tfr = tfr_array_morlet(
            d, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
            output=output, zero_mean=zero_mean, verbose=False,
        )
        out.append(tfr[0])
    return RaggedEpochsTFR(
        out, epochs.info, freqs, epochs.tmin, output=output,
        events=epochs.events, metadata=epochs.metadata, sfreq=sfreq,
    )


def warp_tfr(tfr, landmarks, *, target="median", n_points=None,
             landmark_names=None):
    """Landmark-warp a TFR **in the time-frequency domain**.

    This is what EEGLAB's ``newtimef`` does and what the gait-EEG literature
    means by "linearly time-warped to the group median gait cycle length".
    In ``timefreq.m``::

        timerefs = median(g.timestretch{1}', 2);
        M  = timewarp(marksPos, refsPos);
        TSr = transpose(M * r');            % magnitude
        TStheta = angtimewarp(...);         % phase, circularly
        TStmpall = TSr .* exp(i * TStheta);

    Energy moves along the time axis; the frequency axis is untouched. Contrast
    with warping the signal first, which rescales the time axis the
    oscillations live on and therefore shifts their apparent frequency.
    """
    if isinstance(landmarks, str):
        if tfr.metadata is None or landmarks not in tfr.metadata:
            raise ValueError(f"metadata has no column {landmarks!r}.")
        landmarks = [np.asarray(v, dtype=float) for v in tfr.metadata[landmarks]]
    landmarks = [np.asarray(lm, dtype=float) for lm in landmarks]
    if len(landmarks) != len(tfr):
        raise ValueError(f"Got {len(landmarks)} landmark sets for {len(tfr)} epochs.")

    dst = resolve_target_landmarks(landmarks, target)
    if n_points is None:
        n_points = int(np.median(tfr.lengths))

    warp = _warp_complex if np.iscomplexobj(tfr.get_data(0)) else piecewise_linear_warp
    out = [
        warp(tfr.get_data(i), landmarks[i], dst, n_points, tfr.sfreq)
        for i in range(len(tfr))
    ]

    rec = AlignmentRecord(
        method="piecewise-linear",
        domain="tfr",
        target_coord="seconds",
        original_duration=tfr.lengths / tfr.sfreq,
        original_landmarks=landmarks,
        target_landmarks=dst,
        target_rule=target if isinstance(target, str) else "explicit",
        landmark_names=tuple(landmark_names) if landmark_names else None,
    )
    return RaggedEpochsTFR(
        out, tfr.info, tfr.freqs, float(dst[0]), output=tfr.output,
        alignment=rec, events=tfr.events, metadata=tfr.metadata,
        sfreq=(n_points - 1) / (dst[-1] - dst[0]),
    )
