"""``RaggedEpochs`` -- layer 1 of the architecture.

Stores trials at their true durations. No warping, no padding, no fake
``sfreq``. Everything that needs a common time axis has to ask for one
explicitly (see ``_align.py``).
"""

from __future__ import annotations

import numpy as np

from ._backends import RaggedStore, get_backend
from ._provenance import AlignmentRecord

__all__ = ["RaggedEpochs", "RaggedTimesError"]


class RaggedTimesError(RuntimeError):
    """Raised by ``.times`` when epochs do not share a time axis.

    Deliberately not a warning and deliberately not a silent fallback to the
    shortest common interval. A plausible-but-wrong time axis is the single
    easiest way for this feature to produce quiet scientific errors.
    """


class RaggedEpochs:
    """Epochs of unequal duration.

    Parameters
    ----------
    store : RaggedStore | list of ndarray
        Per-epoch ``(n_channels, n_i)`` data.
    info : mne.Info
        Standard MNE measurement info. ``info["sfreq"]`` stays the *physical*
        sampling frequency and is never overwritten to encode a percent axis.
    tmin : float | ndarray
        Per-epoch start time relative to the event, in seconds. A scalar is
        broadcast.
    events : ndarray, shape (n_epochs, 3), optional
    event_id : dict, optional
    metadata : pandas.DataFrame, optional
    alignment : AlignmentRecord, optional
        Set once an alignment has been applied.
    """

    def __init__(
        self,
        store,
        info,
        tmin=0.0,
        *,
        events=None,
        event_id=None,
        metadata=None,
        alignment=None,
        backend="list",
    ):
        if not isinstance(store, RaggedStore):
            store = get_backend(backend).from_list(list(store))
        self._store = store
        self.info = info
        n = len(store)

        if store.n_channels != len(info["ch_names"]):
            raise ValueError(
                f"store has {store.n_channels} channels but info has "
                f"{len(info['ch_names'])}."
            )

        self._tmin = np.broadcast_to(np.asarray(tmin, dtype=float), (n,)).copy()

        if events is None:
            events = np.c_[np.arange(n), np.zeros(n, int), np.ones(n, int)]
        self.events = np.asarray(events, dtype=int)
        self.event_id = dict(event_id) if event_id else {"epoch": 1}
        self.metadata = metadata
        self.alignment = alignment

    # -- basic properties -----------------------------------------------
    def __len__(self):
        return len(self._store)

    @property
    def sfreq(self) -> float:
        """Physical sampling frequency, in Hz. Never a normalised-phase step."""
        return float(self.info["sfreq"])

    @property
    def ch_names(self) -> list[str]:
        return list(self.info["ch_names"])

    @property
    def lengths(self) -> np.ndarray:
        """Per-epoch sample counts."""
        return self._store.lengths

    @property
    def durations(self) -> np.ndarray:
        """Per-epoch duration in seconds. The experimental data.

        Survives alignment: after ``align_time`` the aligned object still
        reports the durations the trials actually had.
        """
        return self.lengths / self.sfreq

    @property
    def tmin(self) -> np.ndarray:
        """Per-epoch start time relative to the event, in seconds."""
        return self._tmin

    @property
    def tmax(self) -> np.ndarray:
        """Per-epoch end time relative to the event, in seconds."""
        return self._tmin + (self.lengths - 1) / self.sfreq

    @property
    def is_uniform(self) -> bool:
        """True when every epoch shares one time axis."""
        return bool(
            len(np.unique(self.lengths)) == 1
            and np.allclose(self._tmin, self._tmin[0])
        )

    @property
    def times(self):
        """Common time axis -- only when one genuinely exists.

        Raises
        ------
        RaggedTimesError
            When epochs have different durations or different origins.
        """
        if self.is_uniform:
            return self._tmin[0] + np.arange(self.lengths[0]) / self.sfreq
        raise RaggedTimesError(
            f"These {len(self)} epochs do not share a time axis "
            f"(durations {self.durations.min():.3f}-{self.durations.max():.3f} s).\n"
            "There is no single correct answer here, so pick one explicitly:\n"
            "  .durations            per-epoch duration in seconds\n"
            "  .get_times(i)         the time vector of one epoch\n"
            "  .get_times()          all time vectors, ragged\n"
            "  .align_time(...)      produce a common axis (crop / pad /\n"
            "                        normalise / landmark warp), then .times"
        )

    # -- data access ----------------------------------------------------
    def get_times(self, epoch=None):
        """Time vector(s) in seconds.

        With ``epoch=None`` returns a list of per-epoch vectors.
        """
        if epoch is None:
            return [self.get_times(i) for i in range(len(self))]
        return self._tmin[epoch] + np.arange(self.lengths[epoch]) / self.sfreq

    def get_data(self, epoch=None, *, representation="ragged", pad_value=np.nan):
        """Return the data.

        Parameters
        ----------
        epoch : int | None
            One epoch, or all of them.
        representation : {'ragged', 'dense', 'concatenated'}
            There is no default that is right for every caller, so the choice
            is explicit. ``'dense'`` right-pads to ``max(lengths)``;
            ``'concatenated'`` returns ``(n_channels, sum(lengths))``, which is
            what ICA and covariance actually consume.
        """
        if epoch is not None:
            return self._store.get(epoch)
        if representation == "ragged":
            return [self._store.get(i) for i in range(len(self))]
        if representation == "dense":
            return self._store.to_dense(pad_value)
        if representation == "concatenated":
            return np.concatenate([self._store.get(i) for i in range(len(self))], axis=1)
        raise ValueError(
            f"representation must be 'ragged', 'dense' or 'concatenated', "
            f"got {representation!r}"
        )

    # -- selection ------------------------------------------------------
    def __getitem__(self, idx):
        from ._backends import _as_index

        sel = _as_index(idx, len(self))
        md = None
        if self.metadata is not None:
            md = self.metadata.iloc[sel].reset_index(drop=True)
        return RaggedEpochs(
            self._store.select_epochs(sel),
            self.info,
            self._tmin[sel],
            events=self.events[sel],
            event_id=self.event_id,
            metadata=md,
            alignment=self.alignment,
        )

    def pick(self, picks):
        """Select channels by name or index."""
        names = self.ch_names
        if isinstance(picks, str):
            picks = [picks]
        idx = [names.index(p) if isinstance(p, str) else int(p) for p in picks]
        info = self.info.copy()
        with info._unlock():
            pass
        info = _pick_info(info, idx)
        return RaggedEpochs(
            self._store.select_channels(idx),
            info,
            self._tmin,
            events=self.events,
            event_id=self.event_id,
            metadata=self.metadata,
            alignment=self.alignment,
        )

    def copy(self):
        return self[np.arange(len(self))]

    # -- construction helpers -------------------------------------------
    @classmethod
    def from_raw(
        cls,
        raw,
        onsets,
        durations,
        *,
        tmin=0.0,
        picks=None,
        backend="list",
        metadata=None,
        event_id=None,
    ):
        """Extract epochs of individually specified duration from a ``Raw``.

        This is the mne-mobi ``create_gait_cycle_epochs`` pattern -- take each
        natural trial at its actual length -- with the difference that the
        result stays ragged instead of being immediately warped so it can be
        stacked into an ``EpochsArray``.
        """
        import mne

        sfreq = raw.info["sfreq"]
        onsets = np.asarray(onsets, dtype=float)
        durations = np.asarray(durations, dtype=float)
        if onsets.shape != durations.shape:
            raise ValueError("onsets and durations must have the same length.")

        pick_idx = mne.io.pick._picks_to_idx(raw.info, picks, none="data", exclude="bads")
        blocks, keep = [], []
        n_total = len(raw.times)
        for i, (onset, dur) in enumerate(zip(onsets, durations)):
            start = int(round((onset + tmin) * sfreq)) - raw.first_samp
            stop = start + int(round((dur - tmin) * sfreq))
            if start < 0 or stop > n_total:
                continue  # would be TOO_SHORT in mne.Epochs; drop, same as core
            blocks.append(raw.get_data(picks=pick_idx, start=start, stop=stop))
            keep.append(i)

        if not blocks:
            raise RuntimeError("No epoch fitted inside the recording.")

        keep = np.asarray(keep)
        info = _pick_info(raw.info.copy(), pick_idx)
        events = np.c_[
            (onsets[keep] * sfreq).astype(int),
            np.zeros(len(keep), int),
            np.ones(len(keep), int),
        ]
        md = None
        if metadata is not None:
            md = metadata.iloc[keep].reset_index(drop=True)
        return cls(
            get_backend(backend).from_list(blocks),
            info,
            tmin,
            events=events,
            event_id=event_id,
            metadata=md,
        )

    @classmethod
    def from_annotations(cls, raw, description=None, **kwargs):
        """Build from ``raw.annotations``, *honouring* ``duration``.

        ``mne.Epochs`` has accepted ``events=None`` -> ``raw.annotations.onset``
        since 1.7, but its docstring says outright that "the durations of the
        annotations are ignored in this case". The ragged lengths are already
        in the data model and are being deliberately discarded. This reads
        them.
        """
        annot = raw.annotations
        if len(annot) == 0:
            raise ValueError("raw has no annotations.")
        mask = np.ones(len(annot), bool)
        if description is not None:
            wanted = {description} if isinstance(description, str) else set(description)
            mask = np.isin(annot.description, list(wanted))
        if not mask.any():
            raise ValueError(f"No annotation matched description={description!r}.")
        if np.any(annot.duration[mask] <= 0):
            raise ValueError(
                "Some matching annotations have duration == 0, so there is no "
                "epoch length to read. Set Annotations.duration, or pass "
                "explicit durations to RaggedEpochs.from_raw()."
            )
        onsets = annot.onset[mask] - raw.first_time
        return cls.from_raw(raw, onsets, annot.duration[mask], **kwargs)

    # -- repr -----------------------------------------------------------
    def __repr__(self):
        d = self.durations
        span = (
            f"{d[0]:.3f} s"
            if self.is_uniform
            else f"{d.min():.3f}-{d.max():.3f} s (median {np.median(d):.3f})"
        )
        align = f", aligned: {self.alignment.method}" if self.alignment else ""
        return (
            f"<RaggedEpochs | {len(self)} epochs, {self._store.n_channels} channels, "
            f"{span}, {self.sfreq:g} Hz, backend={self._store.name}{align}>"
        )


def _pick_info(info, idx):
    """Subset an Info to the given channel indices."""
    import mne

    return mne.pick_info(info, np.asarray(idx, dtype=int), copy=True, verbose=False)
