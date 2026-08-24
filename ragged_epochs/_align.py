"""Layer 3 -- explicit temporal alignment.

Nothing in this module ever runs implicitly. ``RaggedEpochs.average()`` does
not call it; it raises and points here. That is the design's answer to
agramfort on PR #12315: if a reduction over trials needs a common time axis,
the user has to say which one and accept its consequences.

Two corrections against mne-mobi's implementation are baked in here:

C1  Warping the *signal* before a time-frequency transform shifts apparent
    frequency. EEGLAB's ``newtimef`` warps after the transform -- magnitude by
    matrix multiply, phase by circular interpolation. ``warp_tfr`` does the
    same; ``landmark_warp`` on signal data records
    ``warps_spectral_content=True`` so the choice is at least visible.

C2  The warp target defaults to the **median** landmark latencies across
    trials, as EEGLAB does (``timerefs = median(...)``). mne-mobi's
    ``warp_sig`` used ``np.linspace(0, 1, n_anchors)``, which forces five gait
    anchors to 25/25/25/25 instead of the real stance/swing split of roughly
    60/40. The evenly-spaced behaviour is still reachable as
    ``target="uniform"``, named so the divergence is not silent.
"""

from __future__ import annotations

import numpy as np

from ._container import RaggedEpochs
from ._provenance import AlignmentRecord

__all__ = [
    "piecewise_linear_warp",
    "resolve_target_landmarks",
    "common_crop",
    "pad",
    "duration_normalise",
    "landmark_warp",
    "align_time",
]


# ---------------------------------------------------------------- core warp
def piecewise_linear_warp(data, src_landmarks, dst_landmarks, n_out, sfreq):
    """Warp ``data`` so ``src_landmarks`` land on ``dst_landmarks``.

    Parameters
    ----------
    data : ndarray, shape (..., n_times)
        Warped along the last axis. Works for ``(n_channels, n_times)`` signal
        data and for ``(n_channels, n_freqs, n_times)`` TF data alike.
    src_landmarks, dst_landmarks : array-like of float
        Monotonically increasing landmark times in seconds, both starting at
        the epoch start and ending at the epoch end. Same length.
    n_out : int
        Samples in the output.
    sfreq : float
        Sampling frequency of ``data`` along its last axis.

    Notes
    -----
    Implemented as an *inverse* map: build the uniform output grid, ask which
    source time each output sample comes from, then sample there. This is
    equivalent to EEGLAB's warp matrix and avoids the forward-map-then-invert
    round trip mne-mobi used.
    """
    src = np.asarray(src_landmarks, dtype=float)
    dst = np.asarray(dst_landmarks, dtype=float)
    if src.shape != dst.shape:
        raise ValueError(
            f"src_landmarks {src.shape} and dst_landmarks {dst.shape} must match."
        )
    if src.size < 2:
        raise ValueError("Need at least start and end landmarks.")
    if np.any(np.diff(src) <= 0):
        raise ValueError(f"src_landmarks must be strictly increasing, got {src}.")
    if np.any(np.diff(dst) <= 0):
        raise ValueError(f"dst_landmarks must be strictly increasing, got {dst}.")

    data = np.asarray(data)
    t_src = np.arange(data.shape[-1]) / float(sfreq)
    u = np.linspace(dst[0], dst[-1], n_out)          # uniform in target time
    t = np.interp(u, dst, src)                        # target time -> source time

    flat = data.reshape(-1, data.shape[-1])
    out = np.empty((flat.shape[0], n_out), dtype=flat.dtype)
    for i in range(flat.shape[0]):
        out[i] = np.interp(t, t_src, flat[i])
    return out.reshape(*data.shape[:-1], n_out)


def _warp_complex(data, src, dst, n_out, sfreq):
    """Warp complex TF coefficients: magnitude linearly, phase circularly.

    Linear interpolation of a wrapped phase angle is wrong -- interpolating
    between +179 deg and -179 deg gives 0 deg instead of 180 deg. EEGLAB uses
    ``angtimewarp`` for this; interpolating the unit complex vector and taking
    its argument is the same idea without unwrapping ambiguities.
    """
    mag = piecewise_linear_warp(np.abs(data), src, dst, n_out, sfreq)
    unit = np.exp(1j * np.angle(data))
    re = piecewise_linear_warp(unit.real, src, dst, n_out, sfreq)
    im = piecewise_linear_warp(unit.imag, src, dst, n_out, sfreq)
    return mag * np.exp(1j * np.arctan2(im, re))


def resolve_target_landmarks(landmarks, target="median"):
    """Choose the common landmark latencies every trial is mapped onto.

    Parameters
    ----------
    landmarks : list of ndarray
        Per-epoch landmark times in seconds, all the same length, each
        starting at the epoch start and ending at the epoch end.
    target : {'median', 'mean', 'uniform'} | array-like
        ``'median'`` reproduces EEGLAB's default and is what the gait-EEG
        literature means by "warped to the group median gait cycle".
        ``'uniform'`` spaces the landmarks evenly -- this is what mne-mobi's
        ``warp_sig`` did, and it distorts the stance/swing proportions. It is
        available, but you have to ask for it by name.
    """
    counts = {len(np.atleast_1d(lm)) for lm in landmarks}
    if len(counts) != 1:
        from collections import Counter

        tally = Counter(len(np.atleast_1d(lm)) for lm in landmarks)
        breakdown = "; ".join(
            f"{n} anchors: {k} epochs" for n, k in sorted(tally.items())
        )
        raise ValueError(
            "All epochs must have the same number of landmarks. "
            f"Got {breakdown}.\n"
            "Mapping 4 anchors and 5 anchors onto one target silently aligns "
            "different biomechanical events: a cycle missing LHS would have "
            "its RTO warped onto another cycle's LHS.\n"
            "Either drop the incomplete cycles, or align each anchor-count "
            "group separately and compare them explicitly."
        )
    stacked = np.asarray(landmarks, dtype=float)
    if isinstance(target, str):
        if target == "median":
            out = np.median(stacked, axis=0)
        elif target == "mean":
            out = stacked.mean(axis=0)
        elif target == "uniform":
            out = np.linspace(0.0, float(np.median(stacked[:, -1])), stacked.shape[1])
        else:
            raise ValueError(
                f"target must be 'median', 'mean', 'uniform' or an array, got {target!r}"
            )
    else:
        out = np.asarray(target, dtype=float)
        if out.shape != (stacked.shape[1],):
            raise ValueError(
                f"target has {out.shape} landmarks, epochs have {stacked.shape[1]}."
            )
    if np.any(np.diff(out) <= 0):
        raise ValueError(f"Resolved target landmarks are not increasing: {out}")
    return out


# ------------------------------------------------------------- strategies
def common_crop(epochs: RaggedEpochs) -> RaggedEpochs:
    """Keep only the interval present in every epoch.

    Lossy and honest about it. Appropriate when the tail of the longer trials
    is not the thing being studied; wrong for #5612's mental-arithmetic case,
    where the variable part *is* the process of interest.
    """
    n = int(epochs.lengths.min())
    tmin = float(np.max(epochs.tmin))
    sfreq = epochs.sfreq
    blocks = []
    for i in range(len(epochs)):
        start = int(round((tmin - epochs.tmin[i]) * sfreq))
        blocks.append(epochs.get_data(i)[:, start : start + n])
    rec = AlignmentRecord(
        method="common-crop",
        domain="signal",
        target_coord="seconds",
        original_duration=epochs.durations.copy(),
    )
    return RaggedEpochs(
        blocks, epochs.info, tmin,
        events=epochs.events, event_id=epochs.event_id,
        metadata=epochs.metadata, alignment=rec,
    )


def pad(epochs: RaggedEpochs, pad_value=np.nan):
    """Right-pad every epoch to the longest, keeping all samples.

    Returns
    -------
    epochs : RaggedEpochs
        Uniform, so ``.times`` works.
    nave : ndarray, shape (n_times,)
        How many epochs actually contributed a real sample at each time point.

    The second return value is the point. Under padding, ``nave`` is a
    function of time -- the objection agramfort raised on #12315, where it also
    breaks the noise-covariance scaling used by the inverse. This API hands the
    caller that vector rather than hiding it behind a scalar ``nave``.
    """
    if not np.allclose(epochs.tmin, epochs.tmin[0]):
        raise ValueError(
            "pad() needs a common time origin; epochs have different tmin. "
            "Use align_time(method='landmark') or shift them first."
        )
    n_max = int(epochs.lengths.max())
    dense = epochs.get_data(representation="dense", pad_value=pad_value)
    nave = (np.arange(n_max)[None, :] < epochs.lengths[:, None]).sum(axis=0)
    rec = AlignmentRecord(
        method="pad",
        domain="signal",
        target_coord="seconds",
        original_duration=epochs.durations.copy(),
    )
    out = RaggedEpochs(
        list(dense), epochs.info, float(epochs.tmin[0]),
        events=epochs.events, event_id=epochs.event_id,
        metadata=epochs.metadata, alignment=rec,
    )
    return out, nave


def duration_normalise(epochs: RaggedEpochs, n_points=100) -> RaggedEpochs:
    """Map every epoch's start and end onto a common phase axis.

    The two-landmark special case of ``landmark_warp``. Fine for a cycle with
    no meaningful internal structure; for gait, stimulus-decision-response, or
    reach-to-contact, use the landmarks you have.
    """
    lm = [np.array([0.0, d]) for d in epochs.durations]
    return landmark_warp(epochs, lm, target="median", n_points=n_points,
                         landmark_names=("start", "end"),
                         _method="duration-normalise")


def landmark_warp(
    epochs: RaggedEpochs,
    landmarks,
    *,
    target="median",
    n_points=None,
    landmark_names=None,
    _method="piecewise-linear",
) -> RaggedEpochs:
    """Piecewise-linear warp so per-trial landmarks coincide.

    Parameters
    ----------
    epochs : RaggedEpochs
    landmarks : list of array-like | str
        Per-epoch landmark times in seconds relative to epoch t=0, each
        beginning at the epoch start and ending at the epoch end. Pass a string
        to read a list-column of ``epochs.metadata`` instead -- which is what
        ``Epochs.add_annotations_to_metadata()`` already produces.
    target : {'median', 'mean', 'uniform'} | array-like
        See :func:`resolve_target_landmarks`. Defaults to EEGLAB's median.
    n_points : int | None
        Output samples. Defaults to the median epoch length, so the result
        stays at roughly the native temporal resolution.

    Warnings
    --------
    This warps the **signal**. If the quantity of interest is spectral power,
    warp the time-frequency representation instead (:func:`warp_tfr` in
    ``_tfr.py``) -- stretching a trial before the transform rescales the time
    axis its oscillations live on. The returned object records this in
    ``alignment.warps_spectral_content``.
    """
    if isinstance(landmarks, str):
        if epochs.metadata is None or landmarks not in epochs.metadata:
            raise ValueError(f"metadata has no column {landmarks!r}.")
        landmarks = [np.asarray(v, dtype=float) for v in epochs.metadata[landmarks]]

    landmarks = [np.asarray(lm, dtype=float) for lm in landmarks]
    if len(landmarks) != len(epochs):
        raise ValueError(
            f"Got {len(landmarks)} landmark sets for {len(epochs)} epochs."
        )
    dst = resolve_target_landmarks(landmarks, target)
    if n_points is None:
        n_points = int(np.median(epochs.lengths))

    sfreq = epochs.sfreq
    blocks = [
        piecewise_linear_warp(epochs.get_data(i), landmarks[i], dst, n_points, sfreq)
        for i in range(len(epochs))
    ]

    # The output spans dst[0]..dst[-1] in n_points samples: a real, constant
    # effective rate. No fake sfreq -- info["sfreq"] stays physical and the
    # mapping is recorded in provenance instead.
    info = epochs.info.copy()
    with info._unlock():
        info["sfreq"] = (n_points - 1) / (dst[-1] - dst[0])

    rec = AlignmentRecord(
        method=_method,
        domain="signal",
        target_coord="seconds",
        original_duration=epochs.durations.copy(),
        original_landmarks=landmarks,
        target_landmarks=dst,
        target_rule=target if isinstance(target, str) else "explicit",
        landmark_names=tuple(landmark_names) if landmark_names else None,
    )
    return RaggedEpochs(
        blocks, info, float(dst[0]),
        events=epochs.events, event_id=epochs.event_id,
        metadata=epochs.metadata, alignment=rec,
    )


def align_time(epochs, method="common-crop", **kwargs):
    """Dispatch to an alignment strategy.

    Strategies: ``'common-crop'``, ``'pad'``, ``'duration-normalise'``,
    ``'landmark'``. Signal-domain DTW is deliberately absent from v1 -- unlike
    biomechanical landmarks it derives correspondence from signal similarity,
    so it can align noise and change apparent component durations.
    """
    if method == "common-crop":
        return common_crop(epochs, **kwargs)
    if method == "pad":
        return pad(epochs, **kwargs)
    if method == "duration-normalise":
        return duration_normalise(epochs, **kwargs)
    if method == "landmark":
        return landmark_warp(epochs, **kwargs)
    raise ValueError(
        f"Unknown method {method!r}. Choose from 'common-crop', 'pad', "
        "'duration-normalise', 'landmark'."
    )
